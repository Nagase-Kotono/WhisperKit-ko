import SwiftUI
import Combine
import WhisperKit

// 플랫폼에 따라 UIKit 또는 AppKit을 조건부로 임포트
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

import AVFoundation
import CoreML

// ContentViewModel 클래스는 ObservableObject 프로토콜을 준수하여 SwiftUI 뷰와 데이터 바인딩을 가능하게 함
class ContentViewModel: ObservableObject {
    
    // @Published 속성은 값이 변경될 때마다 뷰에 자동으로 업데이트를 알림
    @Published var confirmedSegments: [TranscriptionSegment] = []          // 확인된 전사 세그먼트 배열
    @Published var unconfirmedSegments: [TranscriptionSegment] = []        // 확인되지 않은 전사 세그먼트 배열
    @Published var bufferEnergy: [Float] = []                              // 추가된 부분: 에너지 버퍼 배열
    @Published var bufferSeconds: Double = 0                               // 추가된 부분: 버퍼된 시간 (초)
    
    // Combine 프레임워크의 AnyCancellable 타입을 저장하기 위한 Set
    private var cancellables = Set<AnyCancellable>()
    
    // 초기화 메서드
    init() {
        // confirmedSegments가 변경될 때마다 0.3초의 디바운스 시간을 적용하여 업데이트
        $confirmedSegments
            .debounce(for: .milliseconds(300), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.notifyScroll()  // 디바운스 후 notifyScroll 메서드 호출
            }
            .store(in: &cancellables)  // 구독을 cancellables에 저장하여 메모리 관리
    }
    
    // 스크롤 이벤트를 처리하기 위한 클로저 변수. 외부에서 할당 가능
    var onScroll: (() -> Void)?
    
    // 스크롤 알림을 트리거하는 메서드
    private func notifyScroll() {
        onScroll?()  // onScroll 클로저가 설정되어 있다면 호출
    }
}

struct ContentView: View {
    
    @StateObject private var viewModel = ContentViewModel()
    
    // MARK: - 상태 변수 선언

    @State private var whisperKit: WhisperKit? = nil  // WhisperKit 인스턴스. 음성 인식 라이브러리 WhisperKit의 사용을 위한 객체.
    #if os(macOS)
    @State private var audioDevices: [AudioDevice]? = nil  // 오디오 장치 목록 (macOS 전용). 사용 가능한 오디오 입력 장치의 목록을 macOS에서만 관리.
    #endif
    @State private var isRecording: Bool = false  // 녹음 중인지 여부. 현재 음성 녹음이 진행 중인지 나타내는 플래그.
    @State private var isTranscribing: Bool = false  // 전사 진행 중인지 여부. 현재 음성 인식을 통해 텍스트로 변환 중인지 나타내는 플래그.
    @State private var currentText: String = ""  // 현재 전사된 텍스트. 실시간으로 텍스트가 여기에 저장됨.
    @State private var currentChunks: [Int: (chunkText: [String], fallbacks: Int)] = [:]  // 현재 처리 중인 청크들. WhisperKit은 음성을 청크 단위로 처리하며, 청크의 텍스트와 폴백 횟수(성공하지 못한 횟수)를 관리.
    @State private var modelStorage: String = "huggingface/models/argmaxinc/whisperkit-coreml"  // 모델 저장 경로. 음성 인식을 위한 모델이 저장된 경로를 지정.

    // MARK: - 모델 관리 변수

    @State private var modelState: ModelState = .unloaded  // 모델 상태. 로드됨 또는 로드되지 않은 상태를 나타냄.
    @State private var localModels: [String] = []  // 로컬에 저장된 모델 목록. 사용 가능한 WhisperKit 모델들을 로컬 파일 경로에서 읽어옴.
    @State private var localModelPath: String = ""  // 로컬 모델 경로. 선택된 로컬 모델의 경로.
    @State private var availableModels: [String] = []  // 사용 가능한 모델 목록. WhisperKit에서 사용할 수 있는 모델의 목록을 저장.
    @State private var availableLanguages: [String] = []  // 사용 가능한 언어 목록. 음성 인식에서 지원되는 언어들의 목록을 저장.
    @State private var disabledModels: [String] = WhisperKit.recommendedModels().disabled  // 사용 불가능한 모델 목록. WhisperKit에서 사용이 권장되지 않거나 지원되지 않는 모델 목록을 저장.

    // MARK: - 사용자 기본 설정 (UserDefaults)

    @AppStorage("selectedAudioInput") private var selectedAudioInput: String = "No Audio Input"  // 선택된 오디오 입력 장치. 사용자가 선택한 오디오 입력 장치 이름을 저장.
    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default  // 선택된 모델. 사용자가 선택한 WhisperKit 모델을 저장.
    @AppStorage("selectedTab") private var selectedTab: String = "Transcribe"  // 선택된 탭 (Transcribe 또는 Stream). 현재 선택된 UI 탭을 저장. 파일 전사 또는 실시간 스트림 중 선택.
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"  // 선택된 작업 (transcribe 또는 translate). 전사 또는 번역 작업 중 사용자가 선택한 작업을 저장.
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"  // 선택된 언어. 음성 인식을 위한 언어를 저장.
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"  // 모델이 저장된 리포지토리 이름. WhisperKit에서 사용할 모델의 Hugging Face 리포지토리 이름을 저장.
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = true  // 타임스탬프 표시 여부. 텍스트 전사 시 타임스탬프를 포함할지 여부를 설정.
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true  // 프롬프트 미리 채우기 사용 여부. 미리 입력된 프롬프트 내용을 사용할지 여부를 설정.
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true  // 캐시 미리 채우기 사용 여부. 캐시 데이터를 미리 채워 빠른 처리를 지원할지 여부를 설정.
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false  // 특수 문자 포함 여부. 전사된 텍스트에 특수 문자를 포함할지 여부를 설정.
    @AppStorage("enableEagerDecoding") private var enableEagerDecoding: Bool = false  // Eager Decoding 사용 여부. 빠르게 디코딩을 시작할지 설정.
    @AppStorage("enableDecoderPreview") private var enableDecoderPreview: Bool = true  // 디코더 미리보기 표시 여부. 디코딩 진행 중 미리보기를 활성화할지 여부를 설정.
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0  // 시작 온도 (디코딩 무작위성 제어). 디코딩 과정에서 무작위성을 제어하는 '온도' 값을 저장.
    @AppStorage("fallbackCount") private var fallbackCount: Double = 5  // 최대 폴백 횟수. 모델이 실패할 경우 다시 시도하는 최대 횟수를 저장.
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 60  // 압축 체크 윈도우 크기. 모델 압축 검사의 기준이 되는 윈도우 크기를 설정.
    @AppStorage("sampleLength") private var sampleLength: Double = 224  // 샘플 길이 (토큰 수). 음성 샘플의 길이를 저장.
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3  // 무음 임계값. 음성 인식에서 무음으로 간주되는 임계값을 설정.
    @AppStorage("useVAD") private var useVAD: Bool = true  // 음성 활동 감지(VAD) 사용 여부. 음성 활동 감지를 사용할지 설정.
    @AppStorage("tokenConfirmationsNeeded") private var tokenConfirmationsNeeded: Double = 2  // 토큰 확인 필요 횟수. 음성 인식에서 토큰이 확인되어야 하는 최소 횟수를 설정.
    @AppStorage("chunkingStrategy") private var chunkingStrategy: ChunkingStrategy = .none  // 청크 전략 (none 또는 vad). 음성 청크 처리 방법을 설정. VAD(Voice Activity Detection)를 사용할지 여부.
    @AppStorage("encoderComputeUnits") private var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine  // 인코더 연산 유닛. CPU와 뉴럴 엔진을 사용할지 여부를 설정.
    @AppStorage("decoderComputeUnits") private var decoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine  // 디코더 연산 유닛. 디코더에 사용할 연산 유닛(CPU와 뉴럴 엔진)을 설정.

    // MARK: - 일반 상태 변수

    @State private var loadingProgressValue: Float = 0.0  // 모델 로딩 진행률. 모델이 로딩되는 중간 상태를 저장.
    @State private var specializationProgressRatio: Float = 0.7  // 모델 특화 진행률 비율. 모델이 특정 언어 또는 작업에 최적화되는 비율을 설정.
    @State private var isFilePickerPresented = false  // 파일 선택기 표시 여부. 파일 선택 UI가 표시 중인지 여부를 저장.
    @State private var firstTokenTime: TimeInterval = 0  // 첫 번째 토큰 생성 시간. 음성 인식 첫 번째 토큰이 생성된 시간을 기록.
    @State private var pipelineStart: TimeInterval = 0  // 파이프라인 시작 시간. 음성 인식 파이프라인이 시작된 시간을 기록.
    @State private var effectiveRealTimeFactor: TimeInterval = 0  // 실시간 비율. 전사된 속도가 실시간 비율로 측정된 값을 저장.
    @State private var effectiveSpeedFactor: TimeInterval = 0  // 속도 계수. 음성 인식의 속도 계수를 저장.
    @State private var totalInferenceTime: TimeInterval = 0  // 전체 추론 시간. 음성 인식에 소요된 총 시간을 저장.
    @State private var tokensPerSecond: TimeInterval = 0  // 초당 토큰 수. 초당 생성된 토큰의 수를 저장.
    @State private var currentLag: TimeInterval = 0  // 현재 지연 시간. 음성 인식 처리에서 발생하는 지연 시간을 기록.
    @State private var currentFallbacks: Int = 0  // 현재 폴백 횟수. 현재 전사 작업 중 실패한 횟수를 저장.
    @State private var currentEncodingLoops: Int = 0  // 현재 인코딩 루프 수. 인코딩이 몇 번 반복되었는지 기록.
    @State private var currentDecodingLoops: Int = 0  // 현재 디코딩 루프 수. 디코딩이 몇 번 반복되었는지 기록.
    @State private var lastBufferSize: Int = 0  // 마지막 버퍼 크기. 마지막으로 처리된 음성 버퍼의 크기를 저장.
    @State private var lastConfirmedSegmentEndSeconds: Float = 0  // 마지막으로 확인된 세그먼트 종료 시간. 마지막 전사된 세그먼트가 끝난 시간을 기록.
    @State private var requiredSegmentsForConfirmation: Int = 4  // 확인에 필요한 세그먼트 수. 확인된 전사 결과에 필요한 최소 세그먼트 수.

    // MARK: - Eager 모드 변수

    @State private var eagerResults: [TranscriptionResult?] = []  // Eager 모드 전사 결과. Eager 모드에서는 결과가 빠르게 제공되며, 여기서 관리됨.
    @State private var prevResult: TranscriptionResult?  // 이전 전사 결과. 마지막으로 전사된 결과를 저장.
    @State private var lastAgreedSeconds: Float = 0.0  // 마지막으로 동의된 시간 (초). 마지막으로 확인된 음성 데이터의 시간이 기록됨.
    @State private var prevWords: [WordTiming] = []  // 이전 단어 타이밍 정보. 이전에 전사된 단어의 타이밍 정보를 저장.
    @State private var lastAgreedWords: [WordTiming] = []  // 마지막으로 동의된 단어들. 마지막으로 확인된 단어 목록을 저장.
    @State private var confirmedWords: [WordTiming] = []  // 확인된 단어들. 전사 작업에서 확인된 단어들을 저장.
    @State private var confirmedText: String = ""  // 확인된 텍스트. 전사 작업에서 확인된 텍스트를 저장.
    @State private var hypothesisWords: [WordTiming] = []  // 가설 단어들. WhisperKit의 가설 단어 타이밍을 저장.
    @State private var hypothesisText: String = ""  // 가설 텍스트. WhisperKit의 가설 텍스트를 저장.

    // MARK: - UI 관련 변수
    
    @State private var showAdvancedSettings = false

    @State private var columnVisibility: NavigationSplitViewVisibility = .all  // 네비게이션 스플릿 뷰의 열 가시성. UI의 네비게이션 영역을 어떻게 표시할지 설정.
    @State private var showComputeUnits: Bool = true  // 연산 유닛 설정 표시 여부. 인코더와 디코더의 연산 유닛 설정을 표시할지 여부를 설정.
    @State private var showAdvancedOptions: Bool = false  // 고급 옵션 표시 여부. 고급 옵션을 표시할지 여부를 설정.
    @State private var transcriptionTask: Task<Void, Never>? = nil  // 전사 작업. 파일 전사 작업을 수행할 비동기 작업을 저장.
    @State private var selectedCategoryId: MenuItem.ID?  // 선택된 메뉴 항목 ID. 사용자 선택된 메뉴 항목의 ID를 저장.
    @State private var transcribeTask: Task<Void, Never>? = nil  // 파일 전사 작업. 파일에서 음성을 텍스트로 변환하는 작업을 저장.

    // 메뉴 항목 구조체
    struct MenuItem: Identifiable, Hashable {
        var id = UUID()
        var name: String
        var image: String
    }

    // 메뉴 항목 배열
    private var menu = [
        MenuItem(name: "Transcribe", image: "book.pages"),  // 파일에서 전사. 파일을 통해 음성을 텍스트로 전사하는 메뉴 항목.
        MenuItem(name: "Stream", image: "waveform.badge.mic"),  // 실시간 스트리밍 전사. 실시간 음성 스트림을 텍스트로 전사하는 메뉴 항목.
    ]

    // 현재 스트림 모드인지 여부를 반환
    private var isStreamMode: Bool {
        self.selectedCategoryId == menu.first(where: { $0.name == "Stream" })?.id
    }

    // MARK: - 컴퓨팅 옵션 반환 함수

    func getComputeOptions() -> ModelComputeOptions {
        return ModelComputeOptions(
            audioEncoderCompute: encoderComputeUnits,
            textDecoderCompute: decoderComputeUnits
        )
    }

    // MARK: - 뷰

    var body: some View {
        // 네비게이션 스플릿 뷰로 사이드바와 상세 뷰를 나눔
        NavigationSplitView(columnVisibility: $columnVisibility) {
            // 사이드바 뷰
            VStack(alignment: .leading) {
                modelSelectorView  // 모델 선택 UI
                    .padding(.vertical)
                computeUnitsView  // 연산 유닛 설정 뷰 (인코더/디코더 설정 UI)
                    .disabled(modelState != .loaded && modelState != .unloaded)  // 모델이 로드되지 않은 경우 비활성화
                    .padding(.bottom)

                // 메뉴 리스트 (Transcribe, Stream 등)
                List(menu, selection: $selectedCategoryId) { item in
                    HStack {
                        Image(systemName: item.image)  // 아이콘 표시
                        Text(item.name)  // 메뉴 항목 이름 표시
                            .font(.system(.title3))
                            .bold()
                    }
                }
                .onChange(of: selectedCategoryId) {
                    // 선택된 메뉴 항목에 따라 탭을 변경
                    selectedTab = menu.first(where: { $0.id == selectedCategoryId })?.name ?? "Transcribe"
                }
                .disabled(modelState != .loaded)  // 모델이 로드되지 않은 경우 메뉴 비활성화
                .foregroundColor(modelState != .loaded ? .secondary : .primary)  // 모델 상태에 따라 메뉴 색상 변경
            }
            .navigationTitle("WhisperAX")  // 사이드바 제목
            .navigationSplitViewColumnWidth(min: 300, ideal: 350)  // 네비게이션 열 크기 설정
            .padding(.horizontal)
            Spacer()
        } detail: {
            // 상세 뷰
            VStack {
                #if os(iOS)
                modelSelectorView  // iOS에서는 모델 선택 뷰를 상세 뷰에 포함
                    .padding()
                #endif
                transcriptionView  // 전사 뷰
                controlsView  // 녹음, 중단 등 제어 버튼을 포함한 컨트롤 뷰
            }
            .toolbar(content: {
                // 상단 도구 모음에 텍스트 복사 버튼 추가
                ToolbarItem {
                    Button {
                        // Eager 디코딩 모드에 따라 전체 전사 결과 복사
                        if (!enableEagerDecoding) {
                            let fullTranscript = formatSegments(viewModel.confirmedSegments + viewModel.unconfirmedSegments, withTimestamps: enableTimestamps).joined(separator: "\n")
                            #if os(iOS)
                            UIPasteboard.general.string = fullTranscript  // iOS에서 복사
                            #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(fullTranscript, forType: .string)  // macOS에서 복사
                            #endif
                        } else {
                            #if os(iOS)
                            UIPasteboard.general.string = confirmedText + hypothesisText
                            #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(confirmedText + hypothesisText, forType: .string)
                            #endif
                        }
                    } label: {
                        Label("Copy Text", systemImage: "doc.on.doc")
                    }
                    .keyboardShortcut("c", modifiers: .command)  // Cmd+C 단축키 추가
                    .foregroundColor(.primary)
                    .frame(minWidth: 0, maxWidth: .infinity)
                }
            })
        }
        .onAppear {
            // 화면이 나타날 때 초기화 작업 수행
            #if os(macOS)
            selectedCategoryId = menu.first(where: { $0.name == selectedTab })?.id  // macOS에서는 선택된 탭을 설정
            #endif
            fetchModels()  // 모델 목록 가져오기
        }
        .environmentObject(viewModel)  // viewModel을 환경 객체로 등록
    }

    // 전사된 텍스트 형식화
    var formattedSegmentsText: String {
        viewModel.confirmedSegments.map { segment in
            let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))] " : ""
            return timestampText + segment.text
        }.joined(separator: "\n")
    }

    // 전사 세그먼트 뷰 (한 줄의 전사 텍스트와 타임스탬프 표시)
    struct TranscriptionSegmentView: View {
        let segment: TranscriptionSegment
        let enableTimestamps: Bool

        var body: some View {
            let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))] " : ""
            Text(timestampText + segment.text)
                .font(.headline)
                .fontWeight(.bold)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // 전사 뷰
    var transcriptionView: some View {
        ScrollViewReader { scrollProxy in
            VStack {
                // VAD 에너지 시각화 (음성 활동 감지)
                if !viewModel.bufferEnergy.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 1) {
                            let startIndex = max(viewModel.bufferEnergy.count - 300, 0)  // 에너지 목록에서 최대 300개 표시
                            ForEach(Array(viewModel.bufferEnergy.enumerated())[startIndex...], id: \.offset) { _, energy in
                                RoundedRectangle(cornerRadius: 2)
                                    .frame(width: 2, height: CGFloat(energy) * 24)  // 에너지 값에 따른 높이 조절
                                    .foregroundColor(energy > Float(silenceThreshold) ? Color.green.opacity(0.2) : Color.red.opacity(0.2))  // 에너지가 무음 임계값을 넘으면 녹색, 아니면 빨간색으로 표시
                            }
                        }
                    }
                    .frame(height: 24)
                    .scrollIndicators(.never)
                }

                if enableEagerDecoding && isStreamMode {
                    // 실시간 스트림 모드일 때 Eager 디코딩 텍스트 표시
                    TextEditor(text: .constant(formattedSegmentsText))
                        .disabled(true)  // 편집 불가
                        .font(.body)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding()
                } else {
                    // 전사된 텍스트를 스크롤 뷰로 표시
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            // 확인된 전사 세그먼트 표시
                            ForEach(viewModel.confirmedSegments) { segment in
                                TranscriptionSegmentView(segment: segment, enableTimestamps: enableTimestamps)
                                    .id(segment.id)  // 고유 ID 부여
                            }
                            // 확인되지 않은 전사 세그먼트 표시 (회색으로 표시)
                            ForEach(viewModel.unconfirmedSegments) { segment in
                                TranscriptionSegmentView(segment: segment, enableTimestamps: enableTimestamps)
                                    .foregroundColor(.gray)
                                    .id(segment.id)  // 고유 ID 부여
                            }
                            // 디코더 미리보기 텍스트 표시 (디코딩 중인 텍스트)
                            if enableDecoderPreview {
                                Text(currentText)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.leading)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding()
                        .onChange(of: viewModel.unconfirmedSegments.count) { _ in
                            // 전사된 세그먼트가 업데이트될 때 마지막 세그먼트로 스크롤 (확인되지 않은)
                            if let lastSegment = viewModel.unconfirmedSegments.last {
                                withAnimation(.easeOut(duration: 0.3)) {
                                    scrollProxy.scrollTo(lastSegment.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .textSelection(.enabled)  // 텍스트 선택 가능
                }

                // 전사 진행 상황 표시
                if let whisperKit,
                   !isStreamMode,
                   isTranscribing,
                   let task = transcribeTask,
                   !task.isCancelled,
                   whisperKit.progress.fractionCompleted < 1
                {
                    HStack {
                        ProgressView(whisperKit.progress)
                            .progressViewStyle(.linear)
                            .labelsHidden()
                            .padding(.horizontal)

                        // 취소 버튼
                        Button {
                            transcribeTask?.cancel()  // 전사 작업 취소
                            transcribeTask = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(BorderlessButtonStyle())
                    }
                }
            }
            .onReceive(viewModel.$unconfirmedSegments) { _ in
                // 전사 세그먼트가 업데이트될 때 마지막 세그먼트로 스크롤
                if let lastSegment = viewModel.unconfirmedSegments.last {
                    withAnimation(.easeOut(duration: 0.3)) {
                        scrollProxy.scrollTo(lastSegment.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - 모델 선택 뷰

    var modelSelectorView: some View {
        VStack {
            HStack {
                Image(systemName: "circle.fill")
                    .foregroundStyle(modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))  // 모델 상태에 따른 색상 변경
                    .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                Text(modelState.description)  // 모델 상태 설명 표시

                Spacer()

                if availableModels.count > 0 {
                    // 사용 가능한 모델을 선택할 수 있는 Picker
                    Picker("", selection: $selectedModel) {
                        ForEach(availableModels, id: \.self) { model in
                            HStack {
                                let modelIcon = localModels.contains { $0 == model.description } ? "checkmark.circle" : "arrow.down.circle.dotted"  // 로컬에 존재하는 모델은 체크 아이콘, 없으면 다운로드 아이콘
                                Text("\(Image(systemName: modelIcon)) \(model.description.components(separatedBy: "_").dropFirst().joined(separator: " "))").tag(model.description)
                            }
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedModel, initial: false) { _, _ in
                        modelState = .unloaded  // 모델을 선택할 때마다 로드 상태로 변경
                    }
                } else {
                    // 모델 목록을 불러오는 동안 로딩 표시
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(0.5)
                }

                // 모델 삭제 버튼
                Button(action: {
                    deleteModel()  // 선택된 모델 삭제
                }, label: {
                    Image(systemName: "trash")
                })
                .help("Delete model")
                .buttonStyle(BorderlessButtonStyle())
                .disabled(localModels.count == 0)  // 로컬에 저장된 모델이 없을 경우 비활성화
                .disabled(!localModels.contains(selectedModel))

                // 모델 폴더 열기 버튼 (macOS 전용)
                #if os(macOS)
                Button(action: {
                    let folderURL = whisperKit?.modelFolder ?? (localModels.contains(selectedModel) ? URL(fileURLWithPath: localModelPath) : nil)
                    if let folder = folderURL {
                        NSWorkspace.shared.open(folder)  // 모델 폴더 열기
                    }
                }, label: {
                    Image(systemName: "folder")
                })
                .buttonStyle(BorderlessButtonStyle())
                #endif

                // 리포지토리 링크 버튼
                Button(action: {
                    if let url = URL(string: "https://huggingface.co/\(repoName)") {
                        #if os(macOS)
                        NSWorkspace.shared.open(url)  // macOS에서는 브라우저로 링크 열기
                        #else
                        UIApplication.shared.open(url)  // iOS에서는 앱 내에서 링크 열기
                        #endif
                    }
                }, label: {
                    Image(systemName: "link.circle")
                })
                .buttonStyle(BorderlessButtonStyle())
            }

            if modelState == .unloaded {
                // 모델 로드 버튼
                Divider()
                Button {
                    resetState()  // 상태 초기화
                    loadModel(selectedModel)  // 모델 로드
                    modelState = .loading  // 로딩 상태로 전환
                } label: {
                    Text("Load Model")
                        .frame(maxWidth: .infinity)
                        .frame(height: 40)
                }
                .buttonStyle(.borderedProminent)
            } else if loadingProgressValue < 1.0 {
                // 모델 로딩 중일 때 진행률 표시
                VStack {
                    HStack {
                        ProgressView(value: loadingProgressValue, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle())
                            .frame(maxWidth: .infinity)

                        Text(String(format: "%.1f%%", loadingProgressValue * 100))  // 진행률 퍼센트로 표시
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    if modelState == .prewarming {
                        Text("Specializing \(selectedModel) for your device...\nThis can take several minutes on first load")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
            }
        }
    }

    // MARK: - 연산 유닛 설정 뷰

    var computeUnitsView: some View {
        // DisclosureGroup: 클릭 시 연산 유닛 설정을 펼쳐서 볼 수 있는 그룹
        DisclosureGroup(isExpanded: $showComputeUnits) {
            VStack(alignment: .leading) {
                // 오디오 인코더 설정
                HStack {
                    Image(systemName: "circle.fill")
                        // 인코더 모델 상태에 따라 원 색상 변경 (로딩 상태에 따라 색상 변화)
                        .foregroundStyle((whisperKit?.audioEncoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        // 모델 상태가 로딩 중일 때 애니메이션 효과 추가
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Audio Encoder")  // 오디오 인코더 레이블
                    Spacer()
                    // 오디오 인코더에 사용할 연산 유닛 선택 Picker (CPU, GPU, Neural Engine 중 선택)
                    Picker("", selection: $encoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)  // CPU만 사용
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)  // CPU와 GPU 혼합 사용
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)  // CPU와 뉴럴 엔진 사용
                    }
                    // 선택된 연산 유닛이 변경되면 모델을 다시 로드하여 반영
                    .onChange(of: encoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)
                    }
                    .pickerStyle(MenuPickerStyle())  // 메뉴 스타일 Picker
                    .frame(width: 150)  // Picker의 너비 설정
                }
                // 텍스트 디코더 설정
                HStack {
                    Image(systemName: "circle.fill")
                        // 디코더 모델 상태에 따라 원 색상 변경
                        .foregroundStyle((whisperKit?.textDecoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Text Decoder")  // 텍스트 디코더 레이블
                    Spacer()
                    // 텍스트 디코더에 사용할 연산 유닛 선택 Picker
                    Picker("", selection: $decoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: decoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)  // 연산 유닛이 변경되면 모델을 다시 로드
                    }
                    .pickerStyle(MenuPickerStyle())  // 메뉴 스타일 Picker
                    .frame(width: 150)  // Picker의 너비 설정
                }
            }
            .padding(.top)  // 위쪽 여백 추가
        } label: {
            // Compute Units 버튼: 클릭 시 위의 DisclosureGroup을 토글
            Button {
                showComputeUnits.toggle()
            } label: {
                Text("Compute Units")
                    .font(.headline)  // 버튼 텍스트 폰트를 제목 스타일로 설정
            }
            .buttonStyle(.plain)  // 버튼 스타일을 기본 Plain 스타일로 설정
        }
    }

    // MARK: - 컨트롤 뷰

    var controlsView: some View {
        VStack {
            basicSettingsView  // 기본 설정 뷰 (기본적인 옵션 설정 UI)

            // 메뉴에서 선택된 항목에 따라 다른 컨트롤 UI를 표시
            if let selectedCategoryId, let item = menu.first(where: { $0.id == selectedCategoryId }) {
                switch item.name {
                    case "Transcribe":
                        // 파일 전사 모드 UI
                        VStack {
                            HStack {
                                // 초기화 버튼: 상태를 초기화하여 전사를 새로 시작
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")  // 초기화 버튼 레이블과 아이콘
                                }
                                .buttonStyle(.borderless)  // 테두리가 없는 버튼 스타일 적용

                                Spacer()

                                audioDevicesView  // 오디오 장치 선택 뷰 (사용 가능한 오디오 입력 장치 목록)

                                Spacer()

                                // 설정 버튼: 고급 옵션을 열기 위한 버튼
                                Button {
                                    showAdvancedOptions.toggle()
                                } label: {
                                    Label("Settings", systemImage: "slider.horizontal.3")  // 설정 아이콘과 레이블
                                }
                                .buttonStyle(.borderless)  // 테두리가 없는 버튼 스타일 적용
                            }

                            HStack {
                                // 파일 선택 버튼 (로딩된 모델에 따라 상태가 다르게 표시)
                                let color: Color = modelState != .loaded ? .gray : .red  // 모델 로드 여부에 따라 색상 설정
                                Button(action: {
                                    withAnimation {
                                        selectFile()  // 파일 선택 애니메이션과 함께 파일 선택 동작 실행
                                    }
                                }) {
                                    Text("FROM FILE")  // 버튼 텍스트
                                        .font(.headline)
                                        .foregroundColor(color)  // 버튼 텍스트 색상
                                        .padding()
                                        .cornerRadius(40)
                                        .frame(minWidth: 70, minHeight: 70)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 40)
                                                .stroke(color, lineWidth: 4)  // 테두리 추가
                                        )
                                }
                                // 파일 선택기 UI를 표시 (오디오 파일만 선택 가능)
                                .fileImporter(
                                    isPresented: $isFilePickerPresented,
                                    allowedContentTypes: [.audio],
                                    allowsMultipleSelection: false,
                                    onCompletion: handleFilePicker  // 파일 선택 완료 후 처리
                                )
                                .lineLimit(1)  // 파일 경로 텍스트 길이 제한
                                .contentTransition(.symbolEffect(.replace))  // 전환 효과
                                .buttonStyle(BorderlessButtonStyle())  // 테두리가 없는 버튼 스타일 적용
                                .disabled(modelState != .loaded)  // 모델이 로드되지 않으면 비활성화
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .padding()

                                // 녹음 버튼
                                ZStack {
                                    Button(action: {
                                        withAnimation {
                                            toggleRecording(shouldLoop: false)  // 녹음 시작/중지 애니메이션 실행
                                        }
                                    }) {
                                        if !isRecording {
                                            // 녹음이 시작되지 않은 경우
                                            Text("RECORD")
                                                .font(.headline)
                                                .foregroundColor(color)
                                                .padding()
                                                .cornerRadius(40)
                                                .frame(minWidth: 70, minHeight: 70)
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 40)
                                                        .stroke(color, lineWidth: 4)
                                                )
                                        } else {
                                            // 녹음 중인 경우 중지 버튼 표시
                                            Image(systemName: "stop.circle.fill")
                                                .resizable()
                                                .scaledToFit()
                                                .frame(width: 70, height: 70)
                                                .padding()
                                                .foregroundColor(modelState != .loaded ? .gray : .red)
                                        }
                                    }
                                    .lineLimit(1)
                                    .contentTransition(.symbolEffect(.replace))
                                    .buttonStyle(BorderlessButtonStyle())
                                    .disabled(modelState != .loaded)  // 모델이 로드되지 않으면 비활성화
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .padding()

                                    if isRecording {
                                        // 녹음 중인 경우 현재 녹음 시간 표시
                                        Text("\(String(format: "%.1f", viewModel.bufferSeconds)) s")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                            .offset(x: 80, y: 0)
                                    }
                                }
                            }
                        }
                    case "Stream":
                        // 실시간 스트리밍 모드 UI
                        VStack {
                            HStack {
                                // 초기화 버튼: 스트리밍을 초기화
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")  // 초기화 아이콘과 레이블
                                }
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .buttonStyle(.borderless)

                                Spacer()

                                audioDevicesView  // 오디오 장치 선택 뷰

                                Spacer()

                                VStack {
                                    // 설정 버튼: 스트리밍 설정을 위한 고급 옵션 버튼
                                    Button {
                                        showAdvancedOptions.toggle()
                                    } label: {
                                        Label("Settings", systemImage: "slider.horizontal.3")  // 설정 아이콘과 레이블
                                    }
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .buttonStyle(.borderless)
                                }
                            }

                            // 녹음 및 스트림 전사 버튼
                            ZStack {
                                Button {
                                    withAnimation {
                                        toggleRecording(shouldLoop: true)  // 스트리밍 녹음 시작/중지
                                    }
                                } label: {
                                    Image(systemName: !isRecording ? "record.circle" : "stop.circle.fill")  // 녹음 중 상태에 따라 버튼 아이콘 변경
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 70, height: 70)
                                        .padding()
                                        .foregroundColor(modelState != .loaded ? .gray : .red)
                                }
                                .contentTransition(.symbolEffect(.replace))
                                .buttonStyle(BorderlessButtonStyle())
                                .disabled(modelState != .loaded)  // 모델이 로드되지 않으면 비활성화
                                .frame(minWidth: 0, maxWidth: .infinity)

                                VStack {
                                    // 인코더 및 디코더 루프 수를 표시
                                    Text("Encoder runs: \(currentEncodingLoops)")
                                        .font(.caption)
                                    Text("Decoder runs: \(currentDecodingLoops)")
                                        .font(.caption)
                                }
                                .offset(x: -120, y: 0)

                                if isRecording {
                                    // 녹음 중인 경우 녹음 시간 표시
                                    Text("\(String(format: "%.1f", viewModel.bufferSeconds)) s")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                        .offset(x: 80, y: 0)
                                }
                            }
                        }
                    default:
                        EmptyView()  // 선택된 항목이 없을 때는 EmptyView를 표시
                }
            }
        }
        .frame(maxWidth: .infinity)  // 뷰의 최대 너비 설정
        .padding(.horizontal)  // 수평 여백 설정
        .sheet(isPresented: $showAdvancedOptions, content: {
            advancedSettingsView  // 고급 설정 뷰를 모달로 표시
                .presentationDetents([.medium, .large])  // 모달 크기 설정
                .presentationBackgroundInteraction(.enabled)
                .presentationContentInteraction(.scrolls)
        })
    }
    
    // MARK: - 오디오 장치 선택 뷰

    // MARK: - 오디오 장치 선택 뷰

    var audioDevicesView: some View {
        Group {
            #if os(macOS)
            HStack {
                // macOS 환경에서 오디오 장치를 선택하는 Picker
                if let audioDevices = audioDevices, audioDevices.count > 0 {
                    Picker("", selection: $selectedAudioInput) {
                        // 사용 가능한 오디오 장치를 나열하고 선택할 수 있도록 함
                        ForEach(audioDevices, id: \.self) { device in
                            Text(device.name).tag(device.name)  // 오디오 장치 이름을 표시하고 태그로 지정
                        }
                    }
                    .frame(width: 250)  // Picker의 너비를 250으로 고정
                    .disabled(isRecording)  // 녹음 중일 때는 장치 선택을 비활성화
                }
            }
            .onAppear {
                // 화면이 나타날 때 AudioProcessor를 사용하여 사용 가능한 오디오 장치를 가져옴
                audioDevices = AudioProcessor.getAudioDevices()
                // 오디오 장치가 존재하고, 선택된 장치가 없을 경우 첫 번째 장치를 기본값으로 설정
                if let audioDevices = audioDevices,
                   !audioDevices.isEmpty,
                   selectedAudioInput == "No Audio Input",
                   let device = audioDevices.first
                {
                    selectedAudioInput = device.name
                }
            }
            #endif
        }
    }

    // MARK: - 기본 설정 뷰

    var basicSettingsView: some View {
        VStack {
            HStack {
                // Transcribe(전사) 또는 Translate(번역) 작업 선택 Picker
                Picker("", selection: $selectedTask) {
                    // DecodingTask에 정의된 모든 작업 옵션을 제공
                    ForEach(DecodingTask.allCases, id: \.self) { task in
                        Text(task.description.capitalized).tag(task.description)
                    }
                }
                // 세그먼트 스타일 Picker로 표시
                .pickerStyle(SegmentedPickerStyle())
                // 모델이 다국어를 지원하지 않는 경우 비활성화
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            }
            .padding(.horizontal)  // 좌우 여백 추가

            // 언어 선택 Picker
            LabeledContent {
                Picker("", selection: $selectedLanguage) {
                    // 사용 가능한 언어 목록을 나열하고 선택 가능하도록 함
                    ForEach(availableLanguages, id: \.self) { language in
                        Text(language.description).tag(language.description)
                    }
                }
                // 모델이 다국어를 지원하지 않으면 비활성화
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            } label: {
                // 언어 선택 Picker에 대한 레이블
                Label("Source Language", systemImage: "globe")
            }
            .padding(.horizontal)  // 좌우 여백 추가
            .padding(.top)  // 상단 여백 추가

            // 성능 관련 메트릭을 표시하는 영역 (실시간 비율, 속도 계수 등)
            HStack {
                // 복잡한 계산식을 미리 처리하여 보기 쉽게 하기 위해 변수로 분리
                let rtfText = "\(effectiveRealTimeFactor.formatted(.number.precision(.fractionLength(3)))) RTF"  // 실시간 비율(RTF)
                let speedFactorText = "\(effectiveSpeedFactor.formatted(.number.precision(.fractionLength(1)))) Speed Factor"  // 속도 계수
                let tokensPerSecondText = "\(tokensPerSecond.formatted(.number.precision(.fractionLength(0)))) tok/s"  // 초당 생성되는 토큰 수
                let firstTokenTimeInterval = firstTokenTime - pipelineStart  // 첫 번째 토큰 생성까지의 시간
                let firstTokenTimeText = "First token: \(firstTokenTimeInterval.formatted(.number.precision(.fractionLength(2))))s"  // 첫 번째 토큰 생성 시간

                // 각 메트릭을 텍스트로 표시
                Text(rtfText)
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()  // 텍스트 사이에 간격 추가
                #if os(macOS)
                Text(speedFactorText)  // 속도 계수 텍스트 (macOS 전용)
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                #endif
                Text(tokensPerSecondText)  // 초당 토큰 수
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                Text(firstTokenTimeText)  // 첫 번째 토큰 생성 시간
                    .font(.system(.body))
                    .lineLimit(1)
            }
            .padding()  // 패딩 추가
            .frame(maxWidth: .infinity)  // 뷰의 최대 너비 설정
        }
    }

//     MARK: - 고급 설정 뷰

//    var advancedSettingsView: some View {
//        #if os(iOS)
//        // iOS에서는 NavigationView 안에 설정 양식을 표시
//        NavigationView {
//            settingsForm
//                .navigationBarTitleDisplayMode(.inline)  // 제목을 상단바에 작게 표시
//        }
//        #else
//        // macOS에서는 고급 설정을 포함한 뷰를 VStack으로 표시
//        VStack {
//            Text("Decoding Options")  // "Decoding Options" 제목 텍스트
//                .font(.title2)  // 제목 폰트 설정
//                .padding()  // 상하 여백 추가
//            settingsForm  // 설정 양식
//                .frame(minWidth: 500, minHeight: 500)  // 최소 크기 설정
//        }
//        #endif
//    }
//
//    // MARK: - 설정 폼
//
//    var settingsForm: some View {
//        List {
//            // 타임스탬프 표시 여부
//            HStack {
//                Text("Show Timestamps")
//                InfoButton("이 옵션을 켜거나 끄면 UI와 자동으로 입력되는 토큰에 타임스탬프가 포함되거나 제외됩니다. \n\n \"Prompt Prefill\" 이 해제되지 않는 한, 이 설정에 따라 <|notimestamps|> 또는 <|0.00|>이 자동으로 적용됩니다.")
//                Spacer()
//                Toggle("", isOn: $enableTimestamps)
//            }
//            .padding(.horizontal)
//
//            // 특수 문자 포함 여부
//            HStack {
//                Text("Special Characters")
//                InfoButton("이 옵션을 켜거나 끄면 음성을 텍스트로 변환할 때 특수 문자(특수 토큰)를(을) 포함할지 여부를 선택할 수 있습니다.")
//                Spacer()
//                Toggle("", isOn: $enableSpecialCharacters)
//            }
//            .padding(.horizontal)
//
//            // 디코더 미리보기 표시 여부
//            HStack {
//                Text("Show Decoder Preview")
//                InfoButton("이 옵션을 켜면 변환된 텍스트 아래에 디코더 출력의 간단한 프리뷰가 UI에 표시됩니다. 디버깅할 때 유용할 수 있습니다.")
//                Spacer()
//                Toggle("", isOn: $enableDecoderPreview)
//            }
//            .padding(.horizontal)
//
//            // 프롬프트 미리 채우기 사용 여부
//            HStack {
//                Text("Prompt Prefill")
//                InfoButton("프롬프트 미리 채우기가 켜져 있으면 디코딩 과정에서 작업, 언어, 타임스탬프 토큰이 자동으로 설정됩니다. \n\n모델이 직접 이 토큰들을 생성하게 하려면 이 옵션을 꺼주세요.")
//                Spacer()
//                Toggle("", isOn: $enablePromptPrefill)
//            }
//            .padding(.horizontal)
//
//            // 캐시 미리 채우기 사용 여부
//            HStack {
//                Text("Cache Prefill")
//                InfoButton("캐시 미리 채우기가 켜져 있으면, 디코더는 디코딩 과정에서 KV 캐시를 매번 계산하는 대신 미리 계산된 캐시 테이블을 사용하려고 시도합니다. \n\n이를 통해 초기 토큰을 강제로 채우는 데 필요한 계산을 건너뛰어 추론 속도를 높일 수 있습니다.")
//                Spacer()
//                Toggle("", isOn: $enableCachePrefill)
//            }
//            .padding(.horizontal)
//
//            // 청크 전략 선택
//            HStack {
//                Text("Chunking Strategy")
//                InfoButton("오디오 데이터를 분할하는 방식을 선택하세요. VAD(음성 활동 감지)를 선택하면, 오디오는 음성이 없는 부분을 기준으로 나누어집니다.")
//                Spacer()
//                Picker("", selection: $chunkingStrategy) {
//                    Text("None").tag(ChunkingStrategy.none)
//                    Text("VAD").tag(ChunkingStrategy.vad)
//                }
//                .pickerStyle(SegmentedPickerStyle())
//            }
//            .padding(.horizontal)
//            .padding(.bottom)
//
//            // 선택한 옵션에 따른 설명을 동적으로 표시
//            if chunkingStrategy == .vad {
//                Text("VAD(음성 활동 감지, Voice Activity Detection)는 음성이 있는 부분과 없는 부분(침묵)을 구분하는 기술입니다. \n\n이 기능은 주로 음성이 언제 시작되고 끝나는지를 판단하여, 녹음된 오디오에서 말하는 구간과 침묵 구간을 분리하는 데 사용됩니다. \n\n예를 들어, 전화 통화나 음성 녹음에서 사람의 말소리가 들리는 부분만 따로 분석하거나 저장하고 싶을 때, VAD는 침묵 구간을 제외하고 말하는 부분만 처리하도록 도와줍니다.")
//                    .font(.footnote)
//                    .foregroundColor(.gray)
//                    .padding(.horizontal)
//                    .transition(.opacity) // 애니메이션으로 설명이 표시되도록 전환
//            }
//
//            // 시작 온도 설정
//            VStack {
//                Text("Starting Temperature:")
//                HStack {
//                    Slider(value: $temperatureStart, in: 0...1, step: 0.1)
//                    Text(temperatureStart.formatted(.number))
//                    InfoButton("디코딩 루프의 초기 무작위성을 제어합니다. 높은 온도는 토큰 선택의 무작위성을 증가시켜 정확도를 향상시킬 수 있습니다.")
//                }
//            }
//            .padding(.horizontal)
//
//            // 최대 폴백 횟수 설정
//            VStack {
//                Text("Max Fallback Count:")
//                HStack {
//                    Slider(value: $fallbackCount, in: 0...5, step: 1)
//                    Text(fallbackCount.formatted(.number))
//                        .frame(width: 30)
//                    InfoButton("디코딩 임계값을 초과했을 때 높은 온도로 폴백할 최대 횟수입니다. 높은 값은 정확도를 높일 수 있지만 속도가 느려질 수 있습니다.")
//                }
//            }
//            .padding(.horizontal)
//
//            // 압축 체크 윈도우 크기 설정
//            VStack {
//                Text("Compression Check Tokens")
//                HStack {
//                    Slider(value: $compressionCheckWindow, in: 0...100, step: 5)
//                    Text(compressionCheckWindow.formatted(.number))
//                        .frame(width: 30)
//                    InfoButton("모델이 반복 루프에 갇혔는지 확인하기 위해 사용할 토큰 수입니다. 낮은 값은 반복을 빨리 감지하지만 너무 낮으면 긴 반복을 놓칠 수 있습니다.")
//                }
//            }
//            .padding(.horizontal)
//
//            // 루프당 최대 토큰 수 설정
//            VStack {
//                Text("Max Tokens Per Loop")
//                HStack {
//                    Slider(value: $sampleLength, in: 0...Double(min(whisperKit?.textDecoder.kvCacheMaxSequenceLength ?? Constants.maxTokenContext, Constants.maxTokenContext)), step: 10)
//                    Text(sampleLength.formatted(.number))
//                        .frame(width: 30)
//                    InfoButton("루프당 생성할 최대 토큰 수입니다. 반복 루프가 너무 길어지는 것을 방지하기 위해 낮출 수 있습니다.")
//                }
//            }
//            .padding(.horizontal)
//
//            // 무음 임계값 설정
//            VStack {
//                Text("Silence Threshold")
//                HStack {
//                    Slider(value: $silenceThreshold, in: 0...1, step: 0.05)
//                    Text(silenceThreshold.formatted(.number))
//                        .frame(width: 30)
//                    InfoButton("오디오의 상대적 무음 임계값입니다. 기준선은 이전 2초 동안의 가장 조용한 100ms로 설정됩니다.")
//                }
//            }
//            .padding(.horizontal)
//
//            // 실험적 설정 섹션
//            Section(header: Text("Experimental")) {
//                // Eager Streaming Mode 사용 여부
//                HStack {
//                    Text("Eager Streaming Mode")
//                    InfoButton("이 옵션을 켜면 전사가 더 자주 업데이트되지만 정확도가 낮아질 수 있습니다.")
//                    Spacer()
//                    Toggle("", isOn: $enableEagerDecoding)
//                }
//                .padding(.horizontal)
//                .padding(.top)
//
//                // 토큰 확인 필요 횟수 설정
//                VStack {
//                    Text("Token Confirmations")
//                    HStack {
//                        Slider(value: $tokenConfirmationsNeeded, in: 1...10, step: 1)
//                        Text(tokenConfirmationsNeeded.formatted(.number))
//                            .frame(width: 30)
//                        InfoButton("스트리밍 과정에서 토큰을 확인하기 위해 필요한 연속 일치 횟수입니다.")
//                    }
//                }
//                .padding(.horizontal)
//            }
//        }
//        .navigationTitle("Decoding Options")
//        .toolbar(content: {
//            ToolbarItem {
//                Button {
//                    showAdvancedOptions = false
//                } label: {
//                    Label("Done", systemImage: "xmark.circle.fill")
//                        .foregroundColor(.primary)
//                }
//            }
//        })
//    }
//
//    // MARK: - 정보 버튼 뷰
//
    struct InfoButton: View {
        var infoText: String
        @State private var showInfo = false

        init(_ infoText: String) {
            self.infoText = infoText
        }

        var body: some View {
            Button(action: {
                self.showInfo = true
            }) {
                Image(systemName: "info.circle")
                    .foregroundColor(.blue)
            }
            .popover(isPresented: $showInfo) {
                Text(infoText)
                    .padding()
            }
            .buttonStyle(BorderlessButtonStyle())
        }
    }
    
    var advancedSettingsView: some View {
        #if os(iOS)
        // iOS에서는 NavigationView 안에 설정 양식을 표시
        NavigationView {
            settingsForm
                .navigationBarTitleDisplayMode(.inline)  // 제목을 상단바에 작게 표시
        }
        #else
        // macOS에서는 고급 설정을 포함한 뷰를 VStack으로 표시
        VStack {
            Text("Decoding Options")  // "Decoding Options" 제목 텍스트
                .font(.title2)  // 제목 폰트 설정
                .padding()  // 상하 여백 추가
            settingsForm  // 설정 양식
                .frame(minWidth: 500, minHeight: 500)  // 최소 크기 설정
        }
        #endif
    }

    // MARK: - 설정 폼

    var settingsForm: some View {
        List {
            // 기본 설정 섹션
            Section(header: Text("Basic Settings")) {
                // 타임스탬프 표시 여부
                ToggleSettingView(text: "Show Timestamps", infoText: "UI와 자동으로 입력되는 토큰에 타임스탬프를 포함하거나 제외합니다.", isOn: $enableTimestamps)
                
                // 특수 문자 포함 여부
                ToggleSettingView(text: "Special Characters", infoText: "음성을 텍스트로 변환할 때 특수 문자를 포함할지 여부를 선택할 수 있습니다.", isOn: $enableSpecialCharacters)
                
                // 디코더 미리보기 표시 여부
                ToggleSettingView(text: "Show Decoder Preview", infoText: "디코더 출력의 간단한 프리뷰를 UI에 표시합니다.", isOn: $enableDecoderPreview)
            }
            
            // 고급 설정 표시 여부
            Section(header: Text("Advanced Settings")) {
                Toggle("Show Advanced Settings", isOn: $showAdvancedSettings)
            }
            
            if showAdvancedSettings {
                // 고급 설정 섹션
                Section(header: Text("Advanced Settings")) {
                    // 프롬프트 미리 채우기
                    ToggleSettingView(text: "Prompt Prefill", infoText: "프롬프트 미리 채우기로 자동 설정된 값을 사용할지 여부를 선택합니다.", isOn: $enablePromptPrefill)
                    
                    // 캐시 미리 채우기
                    ToggleSettingView(text: "Cache Prefill", infoText: "캐시 미리 채우기를 통해 디코딩 속도를 높입니다.", isOn: $enableCachePrefill)
                    
                    // 청크 전략
                    ChunkingStrategyView(chunkingStrategy: $chunkingStrategy)
                    
                    // 시작 온도 설정
                    SliderSettingView(text: "Starting Temperature", value: $temperatureStart, range: 0...1, step: 0.1, infoText: "디코딩 루프의 무작위성을 제어합니다.")
                    
                    // 최대 폴백 횟수
                    SliderSettingView(text: "Max Fallback Count", value: $fallbackCount, range: 0...5, step: 1, infoText: "디코딩 임계값을 초과했을 때 폴백할 최대 횟수입니다.")
                    
                    // 압축 체크 토큰 수
                    SliderSettingView(text: "Compression Check Tokens", value: $compressionCheckWindow, range: 0...100, step: 5, infoText: "모델이 반복 루프에 갇혔는지 확인하기 위한 토큰 수입니다.")
                    
                    // 루프당 최대 토큰 수
                    SliderSettingView(text: "Max Tokens Per Loop", value: $sampleLength, range: 0...Double(min(whisperKit?.textDecoder.kvCacheMaxSequenceLength ?? Constants.maxTokenContext, Constants.maxTokenContext)), step: 10, infoText: "루프당 생성할 최대 토큰 수입니다.")
                    
                    // 무음 임계값
                    SliderSettingView(text: "Silence Threshold", value: $silenceThreshold, range: 0...1, step: 0.05, infoText: "오디오의 상대적 무음 임계값을 설정합니다.")
                }
                
                // 실험적 설정 섹션
                Section(header: Text("Experimental")) {
                    ToggleSettingView(text: "Eager Streaming Mode", infoText: "전사를 더 자주 업데이트하지만 정확도가 낮아질 수 있습니다.", isOn: $enableEagerDecoding)
                    
                    // 토큰 확인 필요 횟수
                    SliderSettingView(text: "Token Confirmations", value: $tokenConfirmationsNeeded, range: 1...10, step: 1, infoText: "스트리밍 과정에서 토큰을 확인하기 위한 연속 일치 횟수입니다.")
                }
            }
        }
        .navigationTitle("Decoding Options")
        .toolbar {
            ToolbarItem {
                Button {
                    showAdvancedOptions = false
                } label: {
                    Label("Done", systemImage: "xmark.circle.fill")
                        .foregroundColor(.primary)
                }
            }
        }
    }

    // MARK: - 공통 설정 뷰 컴포넌트

    struct ToggleSettingView: View {
        var text: String
        var infoText: String
        @Binding var isOn: Bool
        
        var body: some View {
            HStack {
                Text(text)
                InfoButton(infoText)
                Spacer()
                Toggle("", isOn: $isOn)
            }
            .padding(.horizontal)
        }
    }

    struct SliderSettingView: View {
        var text: String
        @Binding var value: Double
        var range: ClosedRange<Double>
        var step: Double
        var infoText: String
        
        var body: some View {
            VStack {
                Text(text)
                HStack {
                    Slider(value: $value, in: range, step: step)
                    Text(value.formatted(.number))
                        .frame(width: 40)
                    InfoButton(infoText)
                }
            }
            .padding(.horizontal)
        }
    }

    struct ChunkingStrategyView: View {
        @Binding var chunkingStrategy: ChunkingStrategy
        
        var body: some View {
            HStack {
                Text("Chunking Strategy")
                InfoButton("오디오 데이터를 분할하는 방식을 선택하세요. VAD(음성 활동 감지)를 선택하면, 오디오는 음성이 없는 부분을 기준으로 나누어집니다.")
                Spacer()
                Picker("", selection: $chunkingStrategy) {
                    Text("None").tag(ChunkingStrategy.none)
                    Text("VAD").tag(ChunkingStrategy.vad)
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            .padding(.horizontal)
        }
    }



    // MARK: - 도우미 함수들
    
    func formatSegments(_ segments: [TranscriptionSegment], withTimestamps: Bool) -> [String] {
        return segments.map { segment in
            let timestampText = withTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))] " : ""
            return timestampText + segment.text
        }
    }

    func fetchModels() {
        availableModels = [selectedModel]

        // First check what's already downloaded
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let modelPath = documents.appendingPathComponent(modelStorage).path

            // Check if the directory exists
            if FileManager.default.fileExists(atPath: modelPath) {
                localModelPath = modelPath
                do {
                    let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
                    for model in downloadedModels where !localModels.contains(model) {
                        localModels.append(model)
                    }
                } catch {
                    print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
                }
            }
        }

        localModels = WhisperKit.formatModelFiles(localModels)
        for model in localModels {
            if !availableModels.contains(model),
               !disabledModels.contains(model)
            {
                availableModels.append(model)
            }
        }

        print("Found locally: \(localModels)")
        print("Previously selected model: \(selectedModel)")

        Task {
            let remoteModels = try await WhisperKit.fetchAvailableModels(from: repoName)
            for model in remoteModels {
                if !availableModels.contains(model),
                   !disabledModels.contains(model)
                {
                    availableModels.append(model)
                }
            }
        }
    }

    func loadModel(_ model: String, redownload: Bool = false) {
        print("Selected Model: \(UserDefaults.standard.string(forKey: "selectedModel") ?? "nil")")
        print("""
            Computing Options:
            - Mel Spectrogram:  \(getComputeOptions().melCompute.description)
            - Audio Encoder:    \(getComputeOptions().audioEncoderCompute.description)
            - Text Decoder:     \(getComputeOptions().textDecoderCompute.description)
            - Prefill Data:     \(getComputeOptions().prefillCompute.description)
        """)

        // 이전 작업 취소 및 상태 초기화
        resetState()

        Task {
            // 이전 모델이 로드된 경우 언로드
            if let existingWhisperKit = whisperKit {
                await existingWhisperKit.unloadModels()
            }

            // 이전 인스턴스 해제
            whisperKit = nil

            // 새로운 WhisperKit 인스턴스 생성
            do {
                whisperKit = try await WhisperKit(
                    computeOptions: getComputeOptions(),
                    verbose: true,
                    logLevel: .debug,
                    prewarm: false,
                    load: false,
                    download: false
                )
                guard let whisperKit = whisperKit else {
                    return
                }

                var folder: URL?

                // 로컬 모델 확인
                if localModels.contains(model), !redownload {
                    folder = URL(fileURLWithPath: localModelPath).appendingPathComponent(model)
                } else {
                    // 모델 다운로드
                    folder = try await WhisperKit.download(
                        variant: model,
                        from: repoName,
                        progressCallback: { progress in
                            DispatchQueue.main.async {
                                loadingProgressValue = Float(progress.fractionCompleted) * specializationProgressRatio
                                modelState = .downloading
                            }
                        }
                    )
                }

                await MainActor.run {
                    loadingProgressValue = specializationProgressRatio
                    modelState = .downloaded
                }

                if let modelFolder = folder {
                    whisperKit.modelFolder = modelFolder

                    await MainActor.run {
                        loadingProgressValue = specializationProgressRatio
                        modelState = .prewarming
                    }

                    let progressBarTask = Task {
                        await updateProgressBar(targetProgress: 0.9, maxTime: 240)
                    }

                    // 모델 프리워밍
                    do {
                        try await whisperKit.prewarmModels()
                        progressBarTask.cancel()
                    } catch {
                        print("Error prewarming models, retrying: \(error.localizedDescription)")
                        progressBarTask.cancel()
                        if !redownload {
                            loadModel(model, redownload: true)
                            return
                        } else {
                            modelState = .unloaded
                            return
                        }
                    }

                    await MainActor.run {
                        loadingProgressValue = specializationProgressRatio + 0.9 * (1 - specializationProgressRatio)
                        modelState = .loading
                    }

                    try await whisperKit.loadModels()

                    await MainActor.run {
                        if !localModels.contains(model) {
                            localModels.append(model)
                        }

                        availableLanguages = Constants.languages.map { $0.key }.sorted()
                        loadingProgressValue = 1.0
                        modelState = whisperKit.modelState
                    }
                }
            } catch {
                print("Error initializing WhisperKit: \(error.localizedDescription)")
            }
        }
    }

    func deleteModel() {
        if localModels.contains(selectedModel) {
            let modelFolder = URL(fileURLWithPath: localModelPath).appendingPathComponent(selectedModel)

            do {
                try FileManager.default.removeItem(at: modelFolder)

                if let index = localModels.firstIndex(of: selectedModel) {
                    localModels.remove(at: index)
                }

                modelState = .unloaded
            } catch {
                print("Error deleting model: \(error)")
            }
        }
    }

    func updateProgressBar(targetProgress: Float, maxTime: TimeInterval) async {
        let initialProgress = loadingProgressValue
        let decayConstant = -log(1 - targetProgress) / Float(maxTime)

        let startTime = Date()

        while true {
            let elapsedTime = Date().timeIntervalSince(startTime)

            // Break down the calculation
            let decayFactor = exp(-decayConstant * Float(elapsedTime))
            let progressIncrement = (1 - initialProgress) * (1 - decayFactor)
            let currentProgress = initialProgress + progressIncrement

            await MainActor.run {
                loadingProgressValue = currentProgress
            }

            if currentProgress >= targetProgress {
                break
            }

            do {
                try await Task.sleep(nanoseconds: 100_000_000)
            } catch {
                break
            }
        }
    }

    func selectFile() {
        isFilePickerPresented = true
    }

    func handleFilePicker(result: Result<[URL], Error>) {
        switch result {
            case let .success(urls):
                guard let selectedFileURL = urls.first else { return }
                if selectedFileURL.startAccessingSecurityScopedResource() {
                    do {
                        // Access the document data from the file URL
                        let audioFileData = try Data(contentsOf: selectedFileURL)

                        // Create a unique file name to avoid overwriting any existing files
                        let uniqueFileName = UUID().uuidString + "." + selectedFileURL.pathExtension

                        // Construct the temporary file URL in the app's temp directory
                        let tempDirectoryURL = FileManager.default.temporaryDirectory
                        let localFileURL = tempDirectoryURL.appendingPathComponent(uniqueFileName)

                        // Write the data to the temp directory
                        try audioFileData.write(to: localFileURL)

                        print("File saved to temporary directory: \(localFileURL)")

                        transcribeFile(path: selectedFileURL.path)
                    } catch {
                        print("File selection error: \(error.localizedDescription)")
                    }
                }
            case let .failure(error):
                print("File selection error: \(error.localizedDescription)")
        }
    }

    func transcribeFile(path: String) {
        resetState()
        whisperKit?.audioProcessor = AudioProcessor()
        self.transcribeTask = Task {
            isTranscribing = true
            do {
                try await transcribeCurrentFile(path: path)
            } catch {
                print("File selection error: \(error.localizedDescription)")
            }
            isTranscribing = false
        }
    }

    func toggleRecording(shouldLoop: Bool) {
        isRecording.toggle()

        if isRecording {
            resetState()
            startRecording(shouldLoop)
        } else {
            stopRecording(shouldLoop)
        }
    }

    func startRecording(_ loop: Bool) {
        if let audioProcessor = whisperKit?.audioProcessor {
            Task(priority: .userInitiated) {
                guard await AudioProcessor.requestRecordPermission() else {
                    print("Microphone access was not granted.")
                    return
                }

                var deviceId: DeviceID?
                #if os(macOS)
                if self.selectedAudioInput != "No Audio Input",
                   let devices = self.audioDevices,
                   let device = devices.first(where: { $0.name == selectedAudioInput })
                {
                    deviceId = device.id
                }

                // There is no built-in microphone
                if deviceId == nil {
                    throw WhisperError.microphoneUnavailable()
                }
                #endif

                try? audioProcessor.startRecordingLive(inputDeviceID: deviceId) { [weak viewModel] _ in
                    DispatchQueue.main.async {
                        viewModel?.bufferEnergy = whisperKit?.audioProcessor.relativeEnergy ?? []
                        viewModel?.bufferSeconds = Double(whisperKit?.audioProcessor.audioSamples.count ?? 0) / Double(WhisperKit.sampleRate)
                    }
                }

                // Delay the timer start by 1 second
                isRecording = true
                isTranscribing = true
                if loop {
                    realtimeLoop()
                }
            }
        }
    }


    func stopRecording(_ loop: Bool) {
        isRecording = false
        stopRealtimeTranscription()
        if let audioProcessor = whisperKit?.audioProcessor {
            audioProcessor.stopRecording()
        }

        // If not looping, transcribe the full buffer
        if !loop {
            self.transcribeTask = Task {
                isTranscribing = true
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                }
                finalizeText()
                isTranscribing = false
            }
        }

        finalizeText()
    }

    func finalizeText() {
        // Finalize unconfirmed text
        Task {
            await MainActor.run {
                if hypothesisText != "" {
                    confirmedText += hypothesisText
                    hypothesisText = ""
                }

                if viewModel.unconfirmedSegments.count > 0 {
                    viewModel.confirmedSegments.append(contentsOf: viewModel.unconfirmedSegments)
                    viewModel.unconfirmedSegments = []
                }
            }
        }
    }

    // MARK: - Transcribe Logic

    func transcribeCurrentFile(path: String) async throws {
        // Load and convert buffer in a limited scope
        let audioFileSamples = try await Task {
            try autoreleasepool {
                let audioFileBuffer = try AudioProcessor.loadAudio(fromPath: path)
                return AudioProcessor.convertBufferToArray(buffer: audioFileBuffer)
            }
        }.value

        let transcription = try await transcribeAudioSamples(audioFileSamples)

        await MainActor.run {
            currentText = ""
            guard let segments = transcription?.segments else {
                return
            }

            self.tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
            self.effectiveRealTimeFactor = transcription?.timings.realTimeFactor ?? 0
            self.effectiveSpeedFactor = transcription?.timings.speedFactor ?? 0
            self.currentEncodingLoops = Int(transcription?.timings.totalEncodingRuns ?? 0)
            self.firstTokenTime = transcription?.timings.firstTokenTime ?? 0
            self.pipelineStart = transcription?.timings.pipelineStart ?? 0
            self.currentLag = transcription?.timings.decodingLoop ?? 0

            // 수정된 부분: viewModel.confirmedSegments로 업데이트
            self.viewModel.confirmedSegments = segments
        }
    }

    func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        let seekClip: [Float] = [lastConfirmedSegmentEndSeconds]

        let options = DecodingOptions(
            verbose: true,
            task: task,
            language: languageCode,
            temperature: Float(temperatureStart),
            temperatureFallbackCount: Int(fallbackCount),
            sampleLength: Int(sampleLength),
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            wordTimestamps: true,
            clipTimestamps: seekClip,
            chunkingStrategy: chunkingStrategy
        )

        // Early stopping checks
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { (progress: TranscriptionProgress) in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                let chunkId = isStreamMode ? 0 : progress.windowId

                // First check if this is a new window for the same chunk, append if so
                var updatedChunk = (chunkText: [progress.text], fallbacks: fallbacks)
                if var currentChunk = self.currentChunks[chunkId], let previousChunkText = currentChunk.chunkText.last {
                    if progress.text.count >= previousChunkText.count {
                        // This is the same window of an existing chunk, so we just update the last value
                        currentChunk.chunkText[currentChunk.chunkText.endIndex - 1] = progress.text
                        updatedChunk = currentChunk
                    } else {
                        // This is either a new window or a fallback (only in streaming mode)
                        if fallbacks == currentChunk.fallbacks && isStreamMode {
                            // New window (since fallbacks havent changed)
                            updatedChunk.chunkText = [updatedChunk.chunkText.first ?? "" + progress.text]
                        } else {
                            // Fallback, overwrite the previous bad text
                            updatedChunk.chunkText[currentChunk.chunkText.endIndex - 1] = progress.text
                            updatedChunk.fallbacks = fallbacks
                            print("Fallback occured: \(fallbacks)")
                        }
                    }
                }

                // Set the new text for the chunk
                self.currentChunks[chunkId] = updatedChunk
                let joinedChunks = self.currentChunks.sorted { $0.key < $1.key }.flatMap { $0.value.chunkText }.joined(separator: "\n")

                self.currentText = joinedChunks
                self.currentFallbacks = fallbacks
                self.currentDecodingLoops += 1
            }

            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    Logging.debug("Early stopping due to compression threshold")
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                Logging.debug("Early stopping due to logprob threshold")
                return false
            }
            return nil
        }

        let transcriptionResults: [TranscriptionResult] = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: options,
            callback: decodingCallback
        )

        let mergedResults = mergeTranscriptionResults(transcriptionResults)

        return mergedResults
    }

    // MARK: Streaming Logic

    func realtimeLoop() {
        transcriptionTask = Task {
            while isRecording && isTranscribing {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                    break
                }
            }
        }
    }

    func stopRealtimeTranscription() {
        isTranscribing = false
        transcriptionTask?.cancel()
    }

    func transcribeCurrentBuffer() async throws {
        guard let whisperKit = whisperKit else { return }

        // Retrieve the current audio buffer from the audio processor
        let currentBuffer = whisperKit.audioProcessor.audioSamples

        // Calculate the size and duration of the next buffer segment
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

        // Only run the transcribe if the next buffer has at least 1 second of audio
        guard nextBufferSeconds > 1 else {
            await MainActor.run {
                if currentText == "" {
                    currentText = "Waiting for speech..."
                }
            }
            try await Task.sleep(nanoseconds: 100_000_000) // sleep for 100ms for next buffer
            return
        }

        if useVAD {
            let voiceDetected = AudioProcessor.isVoiceDetected(
                in: whisperKit.audioProcessor.relativeEnergy,
                nextBufferInSeconds: nextBufferSeconds,
                silenceThreshold: Float(silenceThreshold)
            )
            // Only run the transcribe if the next buffer has voice
            guard voiceDetected else {
                await MainActor.run {
                    if currentText == "" {
                        currentText = "Waiting for speech..."
                    }
                }

                // TODO: Implement silence buffer purging
    //                if nextBufferSeconds > 30 {
    //                    // This is a completely silent segment of 30s, so we can purge the audio and confirm anything pending
    //                    lastConfirmedSegmentEndSeconds = 0
    //                    whisperKit.audioProcessor.purgeAudioSamples(keepingLast: 2 * WhisperKit.sampleRate) // keep last 2s to include VAD overlap
    //                    currentBuffer = whisperKit.audioProcessor.audioSamples
    //                    lastBufferSize = 0
    //                    confirmedSegments.append(contentsOf: unconfirmedSegments)
    //                    unconfirmedSegments = []
    //                }

                // Sleep for 100ms and check the next buffer
                try await Task.sleep(nanoseconds: 100_000_000)
                return
            }
        }

        // Store this for next iterations VAD
        lastBufferSize = currentBuffer.count

        if enableEagerDecoding && isStreamMode {
            // Run realtime transcribe using word timestamps for segmentation
            let transcription = try await transcribeEagerMode(Array(currentBuffer))
            await MainActor.run {
                currentText = ""
                self.tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
                self.firstTokenTime = transcription?.timings.firstTokenTime ?? 0
                self.pipelineStart = transcription?.timings.pipelineStart ?? 0
                self.currentLag = transcription?.timings.decodingLoop ?? 0
                self.currentEncodingLoops = Int(transcription?.timings.totalEncodingRuns ?? 0)

                let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
                self.totalInferenceTime = transcription?.timings.fullPipeline ?? 0
                self.effectiveRealTimeFactor = Double(totalInferenceTime) / totalAudio
                self.effectiveSpeedFactor = totalAudio / Double(totalInferenceTime)
            }
        } else {
            // Run realtime transcribe using timestamp tokens directly
            let transcription = try await transcribeAudioSamples(Array(currentBuffer))

            // We need to run this next part on the main thread
            await MainActor.run {
                currentText = ""
                guard let segments = transcription?.segments else {
                    return
                }

                self.tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
                self.firstTokenTime = transcription?.timings.firstTokenTime ?? 0
                self.pipelineStart = transcription?.timings.pipelineStart ?? 0
                self.currentLag = transcription?.timings.decodingLoop ?? 0
                self.currentEncodingLoops += Int(transcription?.timings.totalEncodingRuns ?? 0)

                let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
                self.totalInferenceTime += transcription?.timings.fullPipeline ?? 0
                self.effectiveRealTimeFactor = Double(totalInferenceTime) / totalAudio
                self.effectiveSpeedFactor = totalAudio / Double(totalInferenceTime)

                // Logic for moving segments to confirmedSegments
                if segments.count > requiredSegmentsForConfirmation {
                    // Calculate the number of segments to confirm
                    let numberOfSegmentsToConfirm = segments.count - requiredSegmentsForConfirmation

                    // Confirm the required number of segments
                    let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
                    let remainingSegments = Array(segments.suffix(requiredSegmentsForConfirmation))

                    // Update lastConfirmedSegmentEnd based on the last confirmed segment
                    if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > lastConfirmedSegmentEndSeconds {
                        lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end
                        print("Last confirmed segment end: \(lastConfirmedSegmentEndSeconds)")

                        // Add confirmed segments to the confirmedSegments array
                        for segment in confirmedSegmentsArray {
                            if !self.viewModel.confirmedSegments.contains(segment) {
                                self.viewModel.confirmedSegments.append(segment)
                            }
                        }
                    }

                    // Update transcriptions to reflect the remaining segments
                    self.viewModel.unconfirmedSegments = remainingSegments
                } else {
                    // Handle the case where segments are fewer or equal to required
                    self.viewModel.unconfirmedSegments = segments
                }
            }
        }
    }

    func transcribeEagerMode(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        guard whisperKit.textDecoder.supportsWordTimestamps else {
            confirmedText = "Eager mode requires word timestamps, which are not supported by the current model: \(selectedModel)."
            return nil
        }

        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        print(selectedLanguage)
        print(languageCode)

        let options = DecodingOptions(
            verbose: true,
            task: task,
            language: languageCode,
            temperature: Float(temperatureStart),
            temperatureFallbackCount: Int(fallbackCount),
            sampleLength: Int(sampleLength),
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            wordTimestamps: true, // required for eager mode
            firstTokenLogProbThreshold: -1.5 // higher threshold to prevent fallbacks from running to often
        )

        // Early stopping checks
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { progress in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                if progress.text.count < currentText.count {
                    if fallbacks == self.currentFallbacks {
                        //                        self.unconfirmedText.append(currentText)
                    } else {
                        print("Fallback occured: \(fallbacks)")
                    }
                }
                self.currentText = progress.text
                self.currentFallbacks = fallbacks
                self.currentDecodingLoops += 1
            }
            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    Logging.debug("Early stopping due to compression threshold")
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                Logging.debug("Early stopping due to logprob threshold")
                return false
            }

            return nil
        }

        Logging.info("[EagerMode] \(lastAgreedSeconds)-\(Double(samples.count) / 16000.0) seconds")

        let streamingAudio = samples
        var streamOptions = options
        streamOptions.clipTimestamps = [lastAgreedSeconds]
        let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
        streamOptions.prefixTokens = lastAgreedTokens
        do {
            let transcription: TranscriptionResult? = try await whisperKit.transcribe(audioArray: streamingAudio, decodeOptions: streamOptions, callback: decodingCallback).first
            await MainActor.run {
                var skipAppend = false
                if let result = transcription {
                    hypothesisWords = result.allWords.filter { $0.start >= lastAgreedSeconds }

                    if let prevResult = prevResult {
                        prevWords = prevResult.allWords.filter { $0.start >= lastAgreedSeconds }
                        let commonPrefix = findLongestCommonPrefix(prevWords, hypothesisWords)
                        Logging.info("[EagerMode] Prev \"\((prevWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Next \"\((hypothesisWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")

                        if commonPrefix.count >= Int(tokenConfirmationsNeeded) {
                            lastAgreedWords = commonPrefix.suffix(Int(tokenConfirmationsNeeded))
                            lastAgreedSeconds = lastAgreedWords.first!.start
                            Logging.info("[EagerMode] Found new last agreed word \"\(lastAgreedWords.first!.word)\" at \(lastAgreedSeconds) seconds")

                            confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - Int(tokenConfirmationsNeeded)))
                            let currentWords = confirmedWords.map { $0.word }.joined()
                            Logging.info("[EagerMode] Current:  \(lastAgreedSeconds) -> \(Double(samples.count) / 16000.0) \(currentWords)")
                        } else {
                            Logging.info("[EagerMode] Using same last agreed time \(lastAgreedSeconds)")
                            skipAppend = true
                        }
                    }
                    prevResult = result
                }

                if !skipAppend {
                    eagerResults.append(transcription)
                }
            }

            await MainActor.run {
                let finalWords = confirmedWords.map { $0.word }.joined()
                confirmedText = finalWords

                // Accept the final hypothesis because it is the last of the available audio
                let lastHypothesis = lastAgreedWords + findLongestDifferentSuffix(prevWords, hypothesisWords)
                hypothesisText = lastHypothesis.map { $0.word }.joined()
            }
        } catch {
            Logging.error("[EagerMode] Error: \(error)")
            finalizeText()
        }


        let mergedResult = mergeTranscriptionResults(eagerResults, confirmedWords: confirmedWords)

        return mergedResult
    }
    
    // 예시로 resetState() 함수 구현
    func resetState() {
        transcribeTask?.cancel()
        isRecording = false
        isTranscribing = false
        whisperKit?.audioProcessor.stopRecording()
        currentText = ""
        currentChunks = [:]

        pipelineStart = Double.greatestFiniteMagnitude
        firstTokenTime = Double.greatestFiniteMagnitude
        effectiveRealTimeFactor = 0
        effectiveSpeedFactor = 0
        totalInferenceTime = 0
        tokensPerSecond = 0
        currentLag = 0
        currentFallbacks = 0
        currentEncodingLoops = 0
        currentDecodingLoops = 0
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        requiredSegmentsForConfirmation = 2
        viewModel.bufferEnergy = []
        viewModel.bufferSeconds = 0

        // 수정된 부분: viewModel의 confirmedSegments와 unconfirmedSegments를 초기화
        viewModel.confirmedSegments = []
        viewModel.unconfirmedSegments = []

        eagerResults = []
        prevResult = nil
        lastAgreedSeconds = 0.0
        prevWords = []
        lastAgreedWords = []
        confirmedWords = []
        confirmedText = ""
        hypothesisWords = []
        hypothesisText = ""
    }
}

#Preview {
    ContentView()
    #if os(macOS)
        .frame(width: 800, height: 500)
    #endif
}

