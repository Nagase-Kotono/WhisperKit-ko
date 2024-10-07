//
//  TranscriptionSegment+Extensions.swift
//  WhisperAX
//
//  Created by Nagase_Kotono on 10/7/24.
//
import WhisperKit
import Foundation

extension TranscriptionSegment: Identifiable {
    public var id: String {
        return "\(start)-\(end)-\(text.hashValue)"
    }
}

extension TranscriptionSegment: Equatable {
    public static func == (lhs: TranscriptionSegment, rhs: TranscriptionSegment) -> Bool {
        return lhs.start == rhs.start && lhs.end == rhs.end && lhs.text == rhs.text
    }
}




