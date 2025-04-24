import os
import threading
import time
import queue
import numpy as np
import torch
import soundcard as sc
import webrtcvad
from faster_whisper import WhisperModel
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, StringVar
import datetime
import wave
import tempfile


class RealtimeWhisperTranslator:
    def __init__(self):
        # 오디오 설정
        self.rate = 16000
        self.chunk_size = 640  # 20ms 프레임 (16000 * 0.02)
        self.frame_duration_ms = 20

        # VAD (Voice Activity Detection) 설정
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # 0~3 중 3으로 변경 (더 엄격한 감지)

        # 음성 버퍼링
        self.voiced_frames = []
        self.silent_frames = 0
        self.max_silent_frames = 25  # 약 0.5초 침묵 (25 * 20ms)
        self.is_speaking = False

        # Whisper 모델 설정
        print("Whisper 모델 로딩 중...")
        # 모델 크기 옵션: "tiny", "base", "small", "medium", "large-v3"
        self.model_size = "tiny"  # 더 빠른 처리를 위해 tiny로 변경
        self.compute_type = "float32"  # 또는 "int8"

        # GPU 사용 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 장치: {self.device}")

        # Whisper 모델 로드
        self.whisper_model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        print("Whisper 모델 로드 완료")

        # 스레드 간 통신을 위한 큐
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        # 종료 플래그
        self.is_running = False

        # 사용 가능한 오디오 장치 목록
        self.output_devices = []
        self.update_device_list()
        self.selected_device = None

        # UI 초기화
        self.init_ui()

    def update_device_list(self):
        """루프백 마이크 목록 가져오기"""
        self.output_devices = []
        try:
            mics = sc.all_microphones(include_loopback=True)
            for idx, mic in enumerate(mics):
                if mic.isloopback:  # 루프백 마이크만 필터링
                    self.output_devices.append((idx, mic.name))
        except Exception as e:
            print(f"오디오 장치 목록 조회 오류: {e}")

    def init_ui(self):
        """UI 초기화"""
        self.window = tk.Tk()
        self.window.title("Whisper 실시간 스피커 음성 번역기")
        self.window.geometry("800x600")

        # 프레임 설정
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # 오디오 장치 선택
        ttk.Label(control_frame, text="오디오 출력 장치:").pack(side=tk.LEFT, padx=5)
        self.device_var = StringVar()
        self.device_dropdown = ttk.Combobox(control_frame, textvariable=self.device_var, state="readonly", width=30)
        self.device_dropdown['values'] = [device[1] for device in self.output_devices]
        if self.output_devices:
            self.device_dropdown.current(0)
        self.device_dropdown.pack(side=tk.LEFT, padx=5)

        # 언어 선택 프레임
        lang_frame = ttk.Frame(self.window)
        lang_frame.pack(fill=tk.X, padx=10, pady=5)

        # 소스 언어 선택
        ttk.Label(lang_frame, text="소스 언어:").pack(side=tk.LEFT, padx=5)
        self.src_lang_var = StringVar(value="ko")
        self.src_lang_dropdown = ttk.Combobox(lang_frame, textvariable=self.src_lang_var, state="readonly", width=10)
        self.src_lang_dropdown['values'] = ["auto", "ko", "en", "ja", "zh", "es", "fr", "de", "ru"]
        self.src_lang_dropdown.pack(side=tk.LEFT, padx=5)

        # 타겟 언어 선택
        ttk.Label(lang_frame, text="번역 언어:").pack(side=tk.LEFT, padx=5)
        self.target_lang_var = StringVar(value="en")
        self.target_lang_dropdown = ttk.Combobox(lang_frame, textvariable=self.target_lang_var, state="readonly", width=10)
        self.target_lang_dropdown['values'] = ["ko", "en", "ja", "zh", "es", "fr", "de", "ru"]
        self.target_lang_dropdown.pack(side=tk.LEFT, padx=5)

        # 모델 설정 프레임
        model_frame = ttk.Frame(self.window)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        # 모델 크기 선택
        ttk.Label(model_frame, text="모델 크기:").pack(side=tk.LEFT, padx=5)
        self.model_size_var = StringVar(value=self.model_size)
        self.model_size_dropdown = ttk.Combobox(model_frame, textvariable=self.model_size_var, state="readonly", width=10)
        self.model_size_dropdown['values'] = ["tiny", "base", "small", "medium", "large-v3"]
        self.model_size_dropdown.pack(side=tk.LEFT, padx=5)
        self.model_size_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)

        # 버튼 프레임
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        # 시작/중지 버튼
        self.start_button = ttk.Button(button_frame, text="시작", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="중지", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 자막 표시 영역
        subtitle_frame = ttk.LabelFrame(self.window, text="자막")
        subtitle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.subtitle_text = scrolledtext.ScrolledText(subtitle_frame, wrap=tk.WORD, font=("Arial", 12))
        self.subtitle_text.pack(fill=tk.BOTH, expand=True)

        # 상태 표시줄
        self.status_var = tk.StringVar()
        self.status_var.set("준비됨")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 윈도우 종료 시 처리
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_model_change(self, event):
        """모델 변경 시 처리"""
        new_model_size = self.model_size_var.get()
        if new_model_size != self.model_size:
            result = messagebox.askquestion("모델 변경", f"모델을 {new_model_size}로 변경하시겠습니까?\n(변경 시 약간의 시간이 소요될 수 있습니다)")
            if result == 'yes':
                self.status_var.set(f"모델 변경 중: {new_model_size}")
                # 스레드로 모델 로딩 실행
                threading.Thread(target=self.load_whisper_model, args=(new_model_size,), daemon=True).start()
            else:
                # 변경 취소
                self.model_size_var.set(self.model_size)

    def load_whisper_model(self, model_size):
        """Whisper 모델 로딩"""
        try:
            # 새 모델 로드
            self.whisper_model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.model_size = model_size
            self.status_var.set(f"모델 변경 완료: {model_size}")
        except Exception as e:
            messagebox.showerror("모델 로딩 오류", f"모델 로딩 실패: {str(e)}")
            self.model_size_var.set(self.model_size)  # 이전 값으로 복원
            self.status_var.set("준비됨")

    def on_closing(self):
        """윈도우 종료 시 처리"""
        self.stop_processing()
        self.window.destroy()

    def start_processing(self):
        if not self.output_devices:
            messagebox.showerror("오류", "사용 가능한 오디오 장치가 없습니다.")
            return

        device_name = self.device_var.get()
        device_id = None
        for d_id, d_name in self.output_devices:
            if d_name == device_name:
                device_id = d_id
                break

        if device_id is None:
            messagebox.showerror("오류", "오디오 장치를 선택해주세요.")
            return

        # 루프백 마이크 설정
        loopback_mics = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
        if not loopback_mics:
            messagebox.showerror("오류", "루프백 마이크를 찾을 수 없습니다. VB-Cable을 설치하거나 VoiceMeeter를 사용하세요.")
            return

        self.selected_device = loopback_mics[device_id]

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("처리 중...")

        self.audio_thread = threading.Thread(target=self.capture_speaker_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        self.recognition_thread = threading.Thread(target=self.process_audio)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()

        self.update_ui()

    def stop_processing(self):
        """오디오 처리 중지"""
        if self.is_running:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("중지됨")

    def capture_speaker_audio(self):
        """스피커 오디오 캡처"""
        try:
            with self.selected_device.recorder(samplerate=self.rate, channels=[0, 1]) as recorder:
                self.status_var.set("스피커 음성 캡처 중...")

                while self.is_running:
                    # 스피커에서 오디오 데이터 캡처
                    data = recorder.record(numframes=self.chunk_size)

                    # 스테레오를 모노로 변환 (채널 평균)
                    mono_data = np.mean(data, axis=1)

                    # 16비트 PCM으로 변환
                    audio_16bit = (mono_data * 32767).astype(np.int16)
                    audio_bytes = audio_16bit.tobytes()

                    # VAD로 음성 활성화 감지
                    try:
                        is_speech = self.vad.is_speech(audio_bytes, self.rate)
                    except:
                        is_speech = False
                        if np.abs(mono_data).mean() > 0.01:  # 임계값 상향 조정
                            is_speech = True

                    if is_speech:
                        # 음성이 감지됨
                        self.voiced_frames.append(audio_bytes)
                        self.silent_frames = 0

                        if not self.is_speaking and len(self.voiced_frames) > 4:  # 약 80ms 음성 감지 후 발화 시작으로 간주
                            self.is_speaking = True
                            self.status_var.set("음성 감지됨")

                    elif self.is_speaking:
                        # 음성 중 잠깐의 침묵 (아직 문장 끝이 아님)
                        self.voiced_frames.append(audio_bytes)
                        self.silent_frames += 1

                        if self.silent_frames >= self.max_silent_frames:
                            # 침묵이 충분히 길면 발화 종료로 간주
                            if len(self.voiced_frames) > 8:  # 노이즈가 아닌 실제 발화인 경우
                                # 음성 데이터를 큐에 전송
                                audio_bytes_concat = b''.join(self.voiced_frames)
                                self.audio_queue.put(audio_bytes_concat)

                            # 상태 초기화
                            self.voiced_frames = []
                            self.silent_frames = 0
                            self.is_speaking = False
                            self.status_var.set("처리 중...")

        except Exception as e:
            print(f"오디오 캡처 오류: {e}")
            self.status_var.set(f"오디오 캡처 오류: {str(e)}")
            self.stop_processing()

    def process_audio(self):
        """오디오 처리 및 Whisper 음성 인식"""
        os.makedirs("temp", exist_ok=True)

        while self.is_running:
            try:
                # 오디오 큐에서 데이터 가져오기
                audio_data = self.audio_queue.get(timeout=1.0)

                # 임시 WAV 파일로 저장
                temp_wav_path = "temp/temp_whisper.wav"
                with wave.open(temp_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16bit = 2 bytes
                    wf.setframerate(self.rate)
                    wf.writeframes(audio_data)

                # 음성 인식 설정
                src_lang = self.src_lang_var.get()
                target_lang = self.target_lang_var.get()

                # Whisper로 음성 인식
                try:
                    self.status_var.set("Whisper 음성 인식 중...")

                    # Whisper 처리
                    language = None if src_lang == "auto" else src_lang

                    # 번역 옵션 설정
                    task = "translate" if target_lang == "en" else "transcribe"

                    # 음성 인식 실행
                    segments, info = self.whisper_model.transcribe(
                        temp_wav_path,
                        task=task,
                        language=language,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                        condition_on_previous_text=True,
                        no_speech_threshold=0.6
                    )

                    # 결과 가져오기
                    detected_lang = info.language
                    transcript = ""

                    for segment in segments:
                        transcript += segment.text

                    # 번역 필요시 (Whisper가 영어로만 번역하므로 다른 언어는 추가 처리 필요)
                    translated = None

                    if transcript and target_lang != "en" and task == "translate":
                        # 영어로 번역된 결과를 다시 타겟 언어로 번역해야 함
                        # 여기서는 구현 생략 (추가 번역 API 필요)
                        pass

                    # 인식된 텍스트가 있으면 큐에 전송
                    if transcript:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        detected_lang_str = f"감지된 언어: {detected_lang}"
                        self.text_queue.put(("화자", transcript, translated, detected_lang_str, timestamp))

                except Exception as e:
                    print(f"Whisper 처리 오류: {e}")
                    self.status_var.set(f"Whisper 처리 오류: {str(e)}")

            except queue.Empty:
                # 큐가 비어있으면 대기
                pass
            except Exception as e:
                print(f"오디오 처리 오류: {e}")

    def update_ui(self):
        """UI 업데이트"""
        try:
            # 텍스트 큐에서 데이터 가져오기
            while not self.text_queue.empty():
                speaker, original_text, translated_text, lang_info, timestamp = self.text_queue.get_nowait()

                # 자막 표시
                self.subtitle_text.configure(state='normal')
                self.subtitle_text.insert(tk.END, f"[{timestamp}] ({lang_info}) ", "small")
                self.subtitle_text.insert(tk.END, f"{original_text}\n", "original")

                # 번역된 텍스트가 있으면 표시
                if translated_text:
                    self.subtitle_text.insert(tk.END, f"→ {translated_text}\n\n", "translated")
                else:
                    self.subtitle_text.insert(tk.END, "\n")

                # 태그 설정
                self.subtitle_text.tag_configure("small", font=("Arial", 8))
                self.subtitle_text.tag_configure("original", font=("Arial", 11))
                self.subtitle_text.tag_configure("translated", font=("Arial", 11, "italic"), foreground="green")

                # 스크롤 최하단으로
                self.subtitle_text.see(tk.END)
                self.subtitle_text.configure(state='disabled')

        except Exception as e:
            print(f"UI 업데이트 오류: {e}")

        # 주기적으로 UI 업데이트
        if self.is_running:
            self.window.after(100, self.update_ui)

    def run(self):
        """애플리케이션 실행"""
        self.window.mainloop()


def main():
    """메인 함수"""
    # 임시 디렉토리 생성
    os.makedirs("temp", exist_ok=True)

    # 번역기 초기화 및 실행
    translator = RealtimeWhisperTranslator()
    translator.run()


if __name__ == "__main__":
    main()
