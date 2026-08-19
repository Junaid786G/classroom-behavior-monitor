"""
pipeline – video processing sub-package.

Modules
-------
ingest      – read frames from file or RTSP stream
detector    – InsightFace face detection wrapper
tracker     – ByteTrack multi-object tracker wrapper
recognizer  – ArcFace embedding extraction + gallery lookup
pose        – MediaPipe pose analysis and head-pose estimation
behavior    – rule-based + ML behavior classifier
processor   – top-level orchestrator wiring all stages together
"""
