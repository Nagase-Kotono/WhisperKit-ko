import SwiftUI
import Combine
import WhisperKit
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import AVFoundation
import CoreML

class ContentViewModel: ObservableObject {
    @Published var confirmedSegments: [TranscriptionSegment] = []
    @Published var unconfirmedSegments: [TranscriptionSegment] = []
    @Published var bufferEnergy: [Float] = []      // 추가된 부분
    @Published var bufferSeconds: Double = 0       // 추가된 부분

    private var cancellables = Set<AnyCancellable>()
    
    init() {
        // Debounce 설정: 0.3초 간격으로 업데이트
        $confirmedSegments
            .debounce(for: .milliseconds(300), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.notifyScroll()
            }
            .store(in: &cancellables)
    }
    
    var onScroll: (() -> Void)?
    
    private func notifyScroll() {
        onScroll?()
    }
}

struct ContentView: View {
    
    @StateObject private var viewModel = ContentViewModel()
    
    // MARK: - 상태 변수 선언

    @State private var whisperKit: WhisperKit? = nil  // WhisperKit 인스턴스
    #if os(macOS)
    @State private var audioDevices: [AudioDevice]? = nil  // 오디오 장치 목록 (macOS 전용)
    #endif
    @State private var isRecording: Bool = false  // 녹음 중인지 여부
    @State private var isTranscribing: Bool = false  // 전사 진행 중인지 여부
    @State private var currentText: String = ""  // 현재 전사된 텍스트
    @State private var currentChunks: [Int: (chunkText: [String], fallbacks: Int)] = [:]  // 현재 처리 중인 청크들
    @State private var modelStorage: String = "huggingface/models/argmaxinc/whisperkit-coreml"  // 모델 저장 경로

    // MARK: - 모델 관리 변수

    @State private var modelState: ModelState = .unloaded  // 모델 상태 (로드됨, 로드되지 않음 등)
    @State private var localModels: [String] = []  // 로컬에 저장된 모델 목록
    @State private var localModelPath: String = ""  // 로컬 모델 경로
    @State private var availableModels: [String] = []  // 사용 가능한 모델 목록
    @State private var availableLanguages: [String] = []  // 사용 가능한 언어 목록
    @State private var disabledModels: [String] = WhisperKit.recommendedModels().disabled  // 사용 불가능한 모델 목록

    // MARK: - 사용자 기본 설정 (UserDefaults)

    @AppStorage("selectedAudioInput") private var selectedAudioInput: String = "No Audio Input"  // 선택된 오디오 입력
    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default  // 선택된 모델
    @AppStorage("selectedTab") private var selectedTab: String = "Transcribe"  // 선택된 탭 (Transcribe 또는 Stream)
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"  // 선택된 작업 (transcribe 또는 translate)
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"  // 선택된 언어
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"  // 모델이 저장된 리포지토리 이름
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = true  // 타임스탬프 표시 여부
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true  // 프롬프트 미리 채우기 사용 여부
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true  // 캐시 미리 채우기 사용 여부
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false  // 특수 문자 포함 여부
    @AppStorage("enableEagerDecoding") private var enableEagerDecoding: Bool = false  // Eager Decoding 사용 여부
    @AppStorage("enableDecoderPreview") private var enableDecoderPreview: Bool = true  // 디코더 미리보기 표시 여부
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0  // 시작 온도 (디코딩 무작위성 제어)
    @AppStorage("fallbackCount") private var fallbackCount: Double = 5  // 최대 폴백 횟수
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 60  // 압축 체크 윈도우 크기
    @AppStorage("sampleLength") private var sampleLength: Double = 224  // 샘플 길이 (토큰 수)
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3  // 무음 임계값
    @AppStorage("useVAD") private var useVAD: Bool = true  // 음성 활동 감지(VAD) 사용 여부
    @AppStorage("tokenConfirmationsNeeded") private var tokenConfirmationsNeeded: Double = 2  // 토큰 확인 필요 횟수
    @AppStorage("chunkingStrategy") private var chunkingStrategy: ChunkingStrategy = .none  // 청크 전략 (none 또는 vad)
    @AppStorage("encoderComputeUnits") private var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine  // 인코더 연산 유닛
    @AppStorage("decoderComputeUnits") private var decoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine  // 디코더 연산 유닛

    // MARK: - 일반 상태 변수

    @State private var loadingProgressValue: Float = 0.0  // 모델 로딩 진행률
    @State private var specializationProgressRatio: Float = 0.7  // 모델 특화 진행률 비율
    @State private var isFilePickerPresented = false  // 파일 선택기 표시 여부
    @State private var firstTokenTime: TimeInterval = 0  // 첫 번째 토큰 생성 시간
    @State private var pipelineStart: TimeInterval = 0  // 파이프라인 시작 시간
    @State private var effectiveRealTimeFactor: TimeInterval = 0  // 실시간 비율
    @State private var effectiveSpeedFactor: TimeInterval = 0  // 속도 계수
    @State private var totalInferenceTime: TimeInterval = 0  // 전체 추론 시간
    @State private var tokensPerSecond: TimeInterval = 0  // 초당 토큰 수
    @State private var currentLag: TimeInterval = 0  // 현재 지연 시간
    @State private var currentFallbacks: Int = 0  // 현재 폴백 횟수
    @State private var currentEncodingLoops: Int = 0  // 현재 인코딩 루프 수
    @State private var currentDecodingLoops: Int = 0  // 현재 디코딩 루프 수
    @State private var lastBufferSize: Int = 0  // 마지막 버퍼 크기
    @State private var lastConfirmedSegmentEndSeconds: Float = 0  // 마지막으로 확인된 세그먼트 종료 시간
    @State private var requiredSegmentsForConfirmation: Int = 4  // 확인에 필요한 세그먼트 수

    // MARK: - Eager 모드 변수

    @State private var eagerResults: [TranscriptionResult?] = []  // Eager 모드 전사 결과
    @State private var prevResult: TranscriptionResult?  // 이전 전사 결과
    @State private var lastAgreedSeconds: Float = 0.0  // 마지막으로 동의된 시간 (초)
    @State private var prevWords: [WordTiming] = []  // 이전 단어 타이밍 정보
    @State private var lastAgreedWords: [WordTiming] = []  // 마지막으로 동의된 단어들
    @State private var confirmedWords: [WordTiming] = []  // 확인된 단어들
    @State private var confirmedText: String = ""  // 확인된 텍스트
    @State private var hypothesisWords: [WordTiming] = []  // 가설 단어들
    @State private var hypothesisText: String = ""  // 가설 텍스트

    // MARK: - UI 관련 변수

    @State private var columnVisibility: NavigationSplitViewVisibility = .all  // 네비게이션 스플릿 뷰의 열 가시성
    @State private var showComputeUnits: Bool = true  // 연산 유닛 설정 표시 여부
    @State private var showAdvancedOptions: Bool = false  // 고급 옵션 표시 여부
    @State private var transcriptionTask: Task<Void, Never>? = nil  // 전사 작업
    @State private var selectedCategoryId: MenuItem.ID?  // 선택된 메뉴 항목 ID
    @State private var transcribeTask: Task<Void, Never>? = nil  // 파일 전사 작업

    // 메뉴 항목 구조체
    struct MenuItem: Identifiable, Hashable {
        var id = UUID()
        var name: String
        var image: String
    }

    // 메뉴 항목 배열
    private var menu = [
        MenuItem(name: "Transcribe", image: "book.pages"),  // 파일에서 전사
        MenuItem(name: "Stream", image: "waveform.badge.mic"),  // 실시간 스트리밍 전사
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
        NavigationSplitView(columnVisibility: $columnVisibility) {
            // 사이드바 뷰
            VStack(alignment: .leading) {
                modelSelectorView  // 모델 선택 뷰
                    .padding(.vertical)
                computeUnitsView  // 연산 유닛 설정 뷰
                    .disabled(modelState != .loaded && modelState != .unloaded)
                    .padding(.bottom)

                // 메뉴 리스트
                List(menu, selection: $selectedCategoryId) { item in
                    HStack {
                        Image(systemName: item.image)
                        Text(item.name)
                            .font(.system(.title3))
                            .bold()
                    }
                }
                .onChange(of: selectedCategoryId) {
                    selectedTab = menu.first(where: { $0.id == selectedCategoryId })?.name ?? "Transcribe"
                }
                .disabled(modelState != .loaded)
                .foregroundColor(modelState != .loaded ? .secondary : .primary)
            }
            .navigationTitle("WhisperAX")
            .navigationSplitViewColumnWidth(min: 300, ideal: 350)
            .padding(.horizontal)
            Spacer()
        } detail: {
            // 상세 뷰
            VStack {
                #if os(iOS)
                modelSelectorView  // 모델 선택 뷰 (iOS에서는 상세 뷰에 표시)
                    .padding()
                #endif
                transcriptionView
                controlsView  // 컨트롤 뷰 (녹음, 설정 등)
            }
            .toolbar(content: {
                ToolbarItem {
                    // 텍스트 복사 버튼
                    Button {
                        if (!enableEagerDecoding) {
                            let fullTranscript = formatSegments(viewModel.confirmedSegments + viewModel.unconfirmedSegments, withTimestamps: enableTimestamps).joined(separator: "\n")
                            #if os(iOS)
                            UIPasteboard.general.string = fullTranscript
                            #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(fullTranscript, forType: .string)
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
                    .keyboardShortcut("c", modifiers: .command)
                    .foregroundColor(.primary)
                    .frame(minWidth: 0, maxWidth: .infinity)
                }
            })
        }
        .onAppear {
            #if os(macOS)
            selectedCategoryId = menu.first(where: { $0.name == selectedTab })?.id  // 초기 선택된 탭 설정
            #endif
            fetchModels()  // 모델 목록 가져오기
        }
        .environmentObject(viewModel)
    }
    
    var formattedSegmentsText: String {
        viewModel.confirmedSegments.map { segment in
            let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))] " : ""
            return timestampText + segment.text
        }.joined(separator: "\n")
    }

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
    
    var transcriptionView: some View {
        ScrollViewReader { scrollProxy in
            VStack {
                // VAD 에너지 시각화
                if !viewModel.bufferEnergy.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 1) {
                            let startIndex = max(viewModel.bufferEnergy.count - 300, 0)
                            ForEach(Array(viewModel.bufferEnergy.enumerated())[startIndex...], id: \.offset) { _, energy in
                                RoundedRectangle(cornerRadius: 2)
                                    .frame(width: 2, height: CGFloat(energy) * 24)
                                    .foregroundColor(energy > Float(silenceThreshold) ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                            }
                        }
                    }
                    .frame(height: 24)
                    .scrollIndicators(.never)
                }

                if enableEagerDecoding && isStreamMode {
                    TextEditor(text: .constant(formattedSegmentsText))
                        .disabled(true)
                        .font(.body)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding()
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(viewModel.confirmedSegments) { segment in
                                TranscriptionSegmentView(segment: segment, enableTimestamps: enableTimestamps)
                                    .id(segment.id) // 고유 ID 부여
                            }
                            ForEach(viewModel.unconfirmedSegments) { segment in
                                TranscriptionSegmentView(segment: segment, enableTimestamps: enableTimestamps)
                                    .foregroundColor(.gray)
                                    .id(segment.id) // 고유 ID 부여
                            }
                            if enableDecoderPreview {
                                Text(currentText)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.leading)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding()
                        .onChange(of: viewModel.confirmedSegments.count) { _ in
                            // confirmedSegments가 업데이트될 때마다 마지막 요소로 스크롤
                            if let lastSegment = viewModel.confirmedSegments.last {
                                withAnimation(.easeOut(duration: 0.3)) {
                                    scrollProxy.scrollTo(lastSegment.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .textSelection(.enabled)
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

                        Button {
                            transcribeTask?.cancel()
                            transcribeTask = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(BorderlessButtonStyle())
                    }
                }
            }
            .onReceive(viewModel.$confirmedSegments) { _ in
                // confirmedSegments가 업데이트될 때마다 마지막 요소로 스크롤
                if let lastSegment = viewModel.confirmedSegments.last {
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
                    .foregroundStyle(modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                    .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                Text(modelState.description)

                Spacer()

                if availableModels.count > 0 {
                    Picker("", selection: $selectedModel) {
                        ForEach(availableModels, id: \.self) { model in
                            HStack {
                                let modelIcon = localModels.contains { $0 == model.description } ? "checkmark.circle" : "arrow.down.circle.dotted"
                                Text("\(Image(systemName: modelIcon)) \(model.description.components(separatedBy: "_").dropFirst().joined(separator: " "))").tag(model.description)
                            }
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedModel, initial: false) { _, _ in
                        modelState = .unloaded  // 모델 상태를 로드되지 않음으로 변경하여 다시 로드 가능하도록 함
                    }
                } else {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(0.5)
                }

                // 모델 삭제 버튼
                Button(action: {
                    deleteModel()
                }, label: {
                    Image(systemName: "trash")
                })
                .help("Delete model")
                .buttonStyle(BorderlessButtonStyle())
                .disabled(localModels.count == 0)
                .disabled(!localModels.contains(selectedModel))

                // 모델 폴더 열기 버튼 (macOS 전용)
                #if os(macOS)
                Button(action: {
                    let folderURL = whisperKit?.modelFolder ?? (localModels.contains(selectedModel) ? URL(fileURLWithPath: localModelPath) : nil)
                    if let folder = folderURL {
                        NSWorkspace.shared.open(folder)
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
                        NSWorkspace.shared.open(url)
                        #else
                        UIApplication.shared.open(url)
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
                    resetState()
                    loadModel(selectedModel)
                    modelState = .loading
                } label: {
                    Text("Load Model")
                        .frame(maxWidth: .infinity)
                        .frame(height: 40)
                }
                .buttonStyle(.borderedProminent)
            } else if loadingProgressValue < 1.0 {
                // 모델 로딩 진행률 표시
                VStack {
                    HStack {
                        ProgressView(value: loadingProgressValue, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle())
                            .frame(maxWidth: .infinity)

                        Text(String(format: "%.1f%%", loadingProgressValue * 100))
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
        DisclosureGroup(isExpanded: $showComputeUnits) {
            VStack(alignment: .leading) {
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle((whisperKit?.audioEncoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Audio Encoder")
                    Spacer()
                    Picker("", selection: $encoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: encoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)  // 연산 유닛이 변경되면 모델을 다시 로드
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle((whisperKit?.textDecoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Text Decoder")
                    Spacer()
                    Picker("", selection: $decoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: decoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)  // 연산 유닛이 변경되면 모델을 다시 로드
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
            }
            .padding(.top)
        } label: {
            Button {
                showComputeUnits.toggle()
            } label: {
                Text("Compute Units")
                    .font(.headline)
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - 컨트롤 뷰

    var controlsView: some View {
        VStack {
            basicSettingsView  // 기본 설정 뷰

            if let selectedCategoryId, let item = menu.first(where: { $0.id == selectedCategoryId }) {
                switch item.name {
                    case "Transcribe":
                        // 파일 전사 모드 컨트롤
                        VStack {
                            HStack {
                                // 초기화 버튼
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")
                                }
                                .buttonStyle(.borderless)

                                Spacer()

                                audioDevicesView  // 오디오 장치 선택 뷰

                                Spacer()

                                // 설정 버튼
                                Button {
                                    showAdvancedOptions.toggle()
                                } label: {
                                    Label("Settings", systemImage: "slider.horizontal.3")
                                }
                                .buttonStyle(.borderless)
                            }

                            HStack {
                                let color: Color = modelState != .loaded ? .gray : .red
                                // 파일 선택 버튼
                                Button(action: {
                                    withAnimation {
                                        selectFile()
                                    }
                                }) {
                                    Text("FROM FILE")
                                        .font(.headline)
                                        .foregroundColor(color)
                                        .padding()
                                        .cornerRadius(40)
                                        .frame(minWidth: 70, minHeight: 70)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 40)
                                                .stroke(color, lineWidth: 4)
                                        )
                                }
                                .fileImporter(
                                    isPresented: $isFilePickerPresented,
                                    allowedContentTypes: [.audio],
                                    allowsMultipleSelection: false,
                                    onCompletion: handleFilePicker
                                )
                                .lineLimit(1)
                                .contentTransition(.symbolEffect(.replace))
                                .buttonStyle(BorderlessButtonStyle())
                                .disabled(modelState != .loaded)
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .padding()

                                // 녹음 버튼
                                ZStack {
                                    Button(action: {
                                        withAnimation {
                                            toggleRecording(shouldLoop: false)
                                        }
                                    }) {
                                        if !isRecording {
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
                                    .disabled(modelState != .loaded)
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .padding()

                                    if isRecording {
                                        // 녹음 시간 표시
                                        Text("\(String(format: "%.1f", viewModel.bufferSeconds)) s")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                            .offset(x: 80, y: 0)
                                    }
                                }
                            }
                        }
                    case "Stream":
                        // 스트림 전사 모드 컨트롤
                        VStack {
                            HStack {
                                // 초기화 버튼
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")
                                }
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .buttonStyle(.borderless)

                                Spacer()

                                audioDevicesView  // 오디오 장치 선택 뷰

                                Spacer()

                                VStack {
                                    // 설정 버튼
                                    Button {
                                        showAdvancedOptions.toggle()
                                    } label: {
                                        Label("Settings", systemImage: "slider.horizontal.3")
                                    }
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .buttonStyle(.borderless)
                                }
                            }

                            // 녹음 및 전사 버튼
                            ZStack {
                                Button {
                                    withAnimation {
                                        toggleRecording(shouldLoop: true)
                                    }
                                } label: {
                                    Image(systemName: !isRecording ? "record.circle" : "stop.circle.fill")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 70, height: 70)
                                        .padding()
                                        .foregroundColor(modelState != .loaded ? .gray : .red)
                                }
                                .contentTransition(.symbolEffect(.replace))
                                .buttonStyle(BorderlessButtonStyle())
                                .disabled(modelState != .loaded)
                                .frame(minWidth: 0, maxWidth: .infinity)

                                VStack {
                                    // 인코더 및 디코더 루프 수 표시
                                    Text("Encoder runs: \(currentEncodingLoops)")
                                        .font(.caption)
                                    Text("Decoder runs: \(currentDecodingLoops)")
                                        .font(.caption)
                                }
                                .offset(x: -120, y: 0)

                                if isRecording {
                                    // 녹음 시간 표시
                                    Text("\(String(format: "%.1f", viewModel.bufferSeconds)) s")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                        .offset(x: 80, y: 0)
                                }
                            }
                        }
                    default:
                        EmptyView()
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal)
        .sheet(isPresented: $showAdvancedOptions, content: {
            advancedSettingsView
                .presentationDetents([.medium, .large])
                .presentationBackgroundInteraction(.enabled)
                .presentationContentInteraction(.scrolls)
        })
    }

    // MARK: - 오디오 장치 선택 뷰

    var audioDevicesView: some View {
        Group {
            #if os(macOS)
            HStack {
                if let audioDevices = audioDevices, audioDevices.count > 0 {
                    Picker("", selection: $selectedAudioInput) {
                        ForEach(audioDevices, id: \.self) { device in
                            Text(device.name).tag(device.name)
                        }
                    }
                    .frame(width: 250)
                    .disabled(isRecording)
                }
            }
            .onAppear {
                audioDevices = AudioProcessor.getAudioDevices()
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
                // 작업 선택 (Transcribe 또는 Translate)
                Picker("", selection: $selectedTask) {
                    ForEach(DecodingTask.allCases, id: \.self) { task in
                        Text(task.description.capitalized).tag(task.description)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            }
            .padding(.horizontal)

            // 언어 선택
            LabeledContent {
                Picker("", selection: $selectedLanguage) {
                    ForEach(availableLanguages, id: \.self) { language in
                        Text(language.description).tag(language.description)
                    }
                }
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            } label: {
                Label("Source Language", systemImage: "globe")
            }
            .padding(.horizontal)
            .padding(.top)

            // 메트릭 표시 (RTF, 속도 계수 등)
            HStack {
                // 복잡한 표현식을 미리 계산하여 변수로 분리합니다.
                let rtfText = "\(effectiveRealTimeFactor.formatted(.number.precision(.fractionLength(3)))) RTF"
                let speedFactorText = "\(effectiveSpeedFactor.formatted(.number.precision(.fractionLength(1)))) Speed Factor"
                let tokensPerSecondText = "\(tokensPerSecond.formatted(.number.precision(.fractionLength(0)))) tok/s"
                let firstTokenTimeInterval = firstTokenTime - pipelineStart
                let firstTokenTimeText = "First token: \(firstTokenTimeInterval.formatted(.number.precision(.fractionLength(2))))s"

                Text(rtfText)
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                #if os(macOS)
                Text(speedFactorText)
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                #endif
                Text(tokensPerSecondText)
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                Text(firstTokenTimeText)
                    .font(.system(.body))
                    .lineLimit(1)
            }
            .padding()
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - 고급 설정 뷰

    var advancedSettingsView: some View {
        #if os(iOS)
        NavigationView {
            settingsForm
                .navigationBarTitleDisplayMode(.inline)
        }
        #else
        VStack {
            Text("Decoding Options")
                .font(.title2)
                .padding()
            settingsForm
                .frame(minWidth: 500, minHeight: 500)
        }
        #endif
    }

    // MARK: - 설정 폼

    var settingsForm: some View {
        List {
            // 타임스탬프 표시 여부
            HStack {
                Text("Show Timestamps")
                InfoButton("이 옵션을 켜면 UI와 프롬프트에 타임스탬프가 포함됩니다. 비활성화하면 <|notimestamps|> 토큰이 강제됩니다.")
                Spacer()
                Toggle("", isOn: $enableTimestamps)
            }
            .padding(.horizontal)

            // 특수 문자 포함 여부
            HStack {
                Text("Special Characters")
                InfoButton("이 옵션을 켜면 전사된 텍스트에 특수 문자가 포함됩니다.")
                Spacer()
                Toggle("", isOn: $enableSpecialCharacters)
            }
            .padding(.horizontal)

            // 디코더 미리보기 표시 여부
            HStack {
                Text("Show Decoder Preview")
                InfoButton("이 옵션을 켜면 UI에 디코더 출력의 미리보기가 표시됩니다. 디버깅에 유용합니다.")
                Spacer()
                Toggle("", isOn: $enableDecoderPreview)
            }
            .padding(.horizontal)

            // 프롬프트 미리 채우기 사용 여부
            HStack {
                Text("Prompt Prefill")
                InfoButton("이 옵션을 켜면 디코딩 루프에서 작업, 언어 및 타임스탬프 토큰이 강제됩니다. 비활성화하면 모델이 직접 생성합니다.")
                Spacer()
                Toggle("", isOn: $enablePromptPrefill)
            }
            .padding(.horizontal)

            // 캐시 미리 채우기 사용 여부
            HStack {
                Text("Cache Prefill")
                InfoButton("이 옵션을 켜면 디코더가 사전 계산된 KV 캐시를 사용하려고 시도합니다. 이를 통해 초기 프리필 토큰에 필요한 계산을 건너뛰어 추론 속도를 높일 수 있습니다.")
                Spacer()
                Toggle("", isOn: $enableCachePrefill)
            }
            .padding(.horizontal)

            // 청크 전략 선택
            HStack {
                Text("Chunking Strategy")
                InfoButton("오디오 데이터를 청크로 분할할 전략을 선택하세요. VAD를 선택하면 무음 부분에서 분할합니다.")
                Spacer()
                Picker("", selection: $chunkingStrategy) {
                    Text("None").tag(ChunkingStrategy.none)
                    Text("VAD").tag(ChunkingStrategy.vad)
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            .padding(.horizontal)
            .padding(.bottom)

            // 시작 온도 설정
            VStack {
                Text("Starting Temperature:")
                HStack {
                    Slider(value: $temperatureStart, in: 0...1, step: 0.1)
                    Text(temperatureStart.formatted(.number))
                    InfoButton("디코딩 루프의 초기 무작위성을 제어합니다. 높은 온도는 토큰 선택의 무작위성을 증가시켜 정확도를 향상시킬 수 있습니다.")
                }
            }
            .padding(.horizontal)

            // 최대 폴백 횟수 설정
            VStack {
                Text("Max Fallback Count:")
                HStack {
                    Slider(value: $fallbackCount, in: 0...5, step: 1)
                    Text(fallbackCount.formatted(.number))
                        .frame(width: 30)
                    InfoButton("디코딩 임계값을 초과했을 때 높은 온도로 폴백할 최대 횟수입니다. 높은 값은 정확도를 높일 수 있지만 속도가 느려질 수 있습니다.")
                }
            }
            .padding(.horizontal)

            // 압축 체크 윈도우 크기 설정
            VStack {
                Text("Compression Check Tokens")
                HStack {
                    Slider(value: $compressionCheckWindow, in: 0...100, step: 5)
                    Text(compressionCheckWindow.formatted(.number))
                        .frame(width: 30)
                    InfoButton("모델이 반복 루프에 갇혔는지 확인하기 위해 사용할 토큰 수입니다. 낮은 값은 반복을 빨리 감지하지만 너무 낮으면 긴 반복을 놓칠 수 있습니다.")
                }
            }
            .padding(.horizontal)

            // 루프당 최대 토큰 수 설정
            VStack {
                Text("Max Tokens Per Loop")
                HStack {
                    Slider(value: $sampleLength, in: 0...Double(min(whisperKit?.textDecoder.kvCacheMaxSequenceLength ?? Constants.maxTokenContext, Constants.maxTokenContext)), step: 10)
                    Text(sampleLength.formatted(.number))
                        .frame(width: 30)
                    InfoButton("루프당 생성할 최대 토큰 수입니다. 반복 루프가 너무 길어지는 것을 방지하기 위해 낮출 수 있습니다.")
                }
            }
            .padding(.horizontal)

            // 무음 임계값 설정
            VStack {
                Text("Silence Threshold")
                HStack {
                    Slider(value: $silenceThreshold, in: 0...1, step: 0.05)
                    Text(silenceThreshold.formatted(.number))
                        .frame(width: 30)
                    InfoButton("오디오의 상대적 무음 임계값입니다. 기준선은 이전 2초 동안의 가장 조용한 100ms로 설정됩니다.")
                }
            }
            .padding(.horizontal)

            // 실험적 설정 섹션
            Section(header: Text("Experimental")) {
                // Eager Streaming Mode 사용 여부
                HStack {
                    Text("Eager Streaming Mode")
                    InfoButton("이 옵션을 켜면 전사가 더 자주 업데이트되지만 정확도가 낮아질 수 있습니다.")
                    Spacer()
                    Toggle("", isOn: $enableEagerDecoding)
                }
                .padding(.horizontal)
                .padding(.top)

                // 토큰 확인 필요 횟수 설정
                VStack {
                    Text("Token Confirmations")
                    HStack {
                        Slider(value: $tokenConfirmationsNeeded, in: 1...10, step: 1)
                        Text(tokenConfirmationsNeeded.formatted(.number))
                            .frame(width: 30)
                        InfoButton("스트리밍 과정에서 토큰을 확인하기 위해 필요한 연속 일치 횟수입니다.")
                    }
                }
                .padding(.horizontal)
            }
        }
        .navigationTitle("Decoding Options")
        .toolbar(content: {
            ToolbarItem {
                Button {
                    showAdvancedOptions = false
                } label: {
                    Label("Done", systemImage: "xmark.circle.fill")
                        .foregroundColor(.primary)
                }
            }
        })
    }

    // MARK: - 정보 버튼 뷰

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

