# =============================================================================
# SYSTEM INSTRUCTIONS
# =============================================================================

SYSTEM_TRANSCRIPTION = (
    "You are a precise audio transcription assistant. Always return valid JSON."
)

SYSTEM_HOOK_DETECTION = (
    "You are a video content strategist expert at identifying attention hooks. "
    "Always return valid JSON."
)

SYSTEM_INTRO_ANALYSIS = (
    "You are a video content strategist. Always return valid JSON."
)

SYSTEM_SEARCH_RERANK = (
    "You are a search relevance expert. Always return valid JSON."
)

SYSTEM_ENTITY_EXTRACTION = (
    "You are a named entity recognition expert. Identify Brands, Locations, "
    "and People mentioned in the transcript or visible in the visuals."
)

SYSTEM_TOPICS_KEYWORDS = (
    "You are a video metadata expert. Extract high-level topics and keywords "
    "from the provided transcript."
)


# =============================================================================
# FRAME ANALYSIS PROMPTS
# =============================================================================

FRAME_ANALYSIS = """Analyze this video frame and provide a detailed description.

Return a JSON object with:
- "description": A 1-2 sentence description of what's happening in the frame
- "scene_type": One of: "talking_head", "b_roll", "text_overlay", "product_shot", "screen_recording", "animation", "other"
- "objects": List of key objects/elements visible (max 5)
- "text_visible": Any text visible in the frame (or null if none)
- "emotion": If a person is visible, their apparent emotion (or null)
- "visual_hook": Boolean - is this frame visually attention-grabbing?
- "hook_reason": If visual_hook is true, briefly explain why

Example:
{
  "description": "A person speaking directly to camera with an excited expression",
  "scene_type": "talking_head",
  "objects": ["person", "microphone", "ring light"],
  "text_visible": null,
  "emotion": "excited",
  "visual_hook": true,
  "hook_reason": "Strong emotion and direct eye contact"
}

Return ONLY the JSON object, no other text."""


FRAME_BATCH_ANALYSIS = """Analyze these video frames in sequence. They are from the same video, shown in chronological order.

For each frame, provide:
- "frame_index": The frame number (0-indexed)
- "description": What's happening in this frame
- "scene_type": One of: "talking_head", "b_roll", "text_overlay", "product_shot", "screen_recording", "animation", "other"
- "objects": Key objects visible (max 3)
- "visual_hook": Boolean - is this frame attention-grabbing?
- "transition": Did the scene change significantly from the previous frame?

Return a JSON array of analysis objects, one per frame.
Return ONLY the JSON array, no other text."""


# =============================================================================
# TRANSCRIPTION PROMPTS
# =============================================================================

TRANSCRIPTION = """Transcribe this audio with timestamps. Return a JSON array of segments.

Each segment should have:
- "start": start time in seconds (float)
- "end": end time in seconds (float)
- "text": the transcribed text

Example format:
[
  {"start": 0.0, "end": 5.2, "text": "Hello and welcome to this video"},
  {"start": 5.2, "end": 10.1, "text": "Today we're going to talk about..."}
]

Important:
- Create segments at natural speech boundaries (sentences, pauses)
- Each segment should be 3-10 seconds long
- Include all spoken words accurately
- Return ONLY the JSON array, no other text"""


# =============================================================================
# SCENE DETECTION PROMPTS
# =============================================================================

def scene_detection(frame_info: str) -> str:
    """Generate scene detection prompt with frame timestamps."""
    return f"""Analyze this sequence of video frames to identify distinct scenes.

FRAME TIMESTAMPS:
{frame_info}

A 'scene' is a coherent segment where the content, setting, or topic remains consistent.
Look for significant changes in:
- Location or setting
- Visual style (e.g., switch from talking head to B-roll footage)
- Topic or activity being shown
- Camera angle or shot type

For each scene, provide:
- "start_time": The timestamp (in seconds) where this scene begins
- "end_time": The timestamp (in seconds) where this scene ends
- "label": A short title (3-5 words) describing the scene type
- "description": A detailed 1-2 sentence description of what is happening in this scene

Return a JSON object with a "scenes" array:
{{
  "scenes": [
    {{
      "start_time": 0.0,
      "end_time": 45.0,
      "label": "Introduction",
      "description": "The host greets viewers from a home office setup, introducing today's topic."
    }}
  ]
}}

Important:
- Use the ACTUAL timestamps from the list above, not frame indices
- Ensure scenes are contiguous (end_time of one scene should match start_time of next)
- Provide meaningful descriptions that explain WHAT is happening
- Return ONLY the JSON object."""


def scene_detection_simple(frame_info: str) -> str:
    """Simplified scene detection prompt for providers with limited image support."""
    return f"""Analyze these video frames to identify distinct scenes.

FRAME TIMESTAMPS:
{frame_info}

For each scene, provide:
- "start_time": Timestamp in seconds where scene begins
- "end_time": Timestamp in seconds where scene ends
- "label": Short scene title (3-5 words)
- "description": 1-2 sentence description of what's happening

Return JSON: {{"scenes": [...]}}

Use the ACTUAL timestamps above. Return ONLY the JSON object."""


# =============================================================================
# ENTITY EXTRACTION PROMPTS
# =============================================================================

def entity_extraction(transcript: str) -> str:
    """Generate entity extraction prompt with transcript."""
    return f"""Transcript:
{transcript}

Identify unique entities (Brands, Locations, People) mentioned in the transcript or visible in the frames.

Return a JSON list of objects:
- "name": Name of the entity
- "type": One of "Brand", "Location", "Person"
- "count": How many times it was mentioned/seen
- "description": Context or description

Return ONLY the JSON list."""


def entity_extraction_detailed(transcript: str) -> str:
    """Detailed entity extraction prompt with examples."""
    return f"""Transcript:
{transcript}

Identify unique entities mentioned:
- Brands (e.g., Apple, Nike, YouTube)
- Locations (e.g., New York, The Studio, California)
- People (e.g., Steve Jobs, The Host, MrBeast)

Return a JSON list of objects, each with:
- "name": Name of the entity
- "type": One of "Brand", "Location", "Person"
- "count": How many times it was mentioned/seen
- "description": A brief 1-sentence description or context

Example:
[
  {{"name": "Tesla", "type": "Brand", "count": 3, "description": "Electric vehicle manufacturer mentioned during the intro."}}
]

Return ONLY the JSON list."""


# =============================================================================
# TOPICS AND KEYWORDS PROMPTS
# =============================================================================

def topics_keywords(transcript: str) -> str:
    """Generate topics/keywords extraction prompt."""
    return f"""Transcript:
{transcript}

Return a JSON object with:
- "topics": List of high-level topics discussed (max 5, each with "name" and "confidence")
- "keywords": List of important keywords (max 10, each with "text" and "confidence")

Return ONLY the JSON object."""


def topics_keywords_detailed(transcript: str) -> str:
    """Detailed topics/keywords prompt with examples."""
    return f"""Transcript:
{transcript}

Return a JSON object with:
- "topics": List of high-level topics discussed (max 5, each with "name" and "confidence")
- "keywords": List of important keywords (max 10, each with "text" and "confidence")

Example:
{{
  "topics": [
    {{"name": "Product Review", "confidence": 0.95}},
    {{"name": "Tech Comparison", "confidence": 0.8}}
  ],
  "keywords": [
    {{"text": "iPhone 15", "confidence": 1.0}},
    {{"text": "Camera quality", "confidence": 0.9}}
  ]
}}

Return ONLY the JSON object."""


# =============================================================================
# OCR PROMPTS
# =============================================================================

OCR = """Identify ALL text visible in this image. For each text snippet, provide the text and its approximate location.

Return a JSON list of objects:
- "text": The detected text
- "left": X coordinate (0.0 to 1.0)
- "top": Y coordinate (0.0 to 1.0)
- "width": Relative width (0.0 to 1.0)
- "height": Relative height (0.0 to 1.0)

Return ONLY the JSON list."""


# =============================================================================
# HOOK DETECTION PROMPTS
# =============================================================================

def hook_detection(transcript: str, frame_analysis: str) -> str:
    """Generate hook detection prompt with content."""
    return f"""Analyze this video content for attention hooks - moments designed to capture and retain viewer attention.

TRANSCRIPT (with timestamps):
{transcript}

FRAME ANALYSIS:
{frame_analysis}

Identify hooks in these categories:

1. **Visual Hooks**: Scene changes, text overlays, face close-ups, motion spikes
2. **Audio Hooks**: Energy shifts, music drops, silence-to-speech, questions to viewer
3. **Content Patterns**:
   - Pattern interrupts ("But here's the thing...", "Plot twist...")
   - Open loops / cliffhangers ("What happens next will surprise you")
   - Contrast and surprise elements
   - Direct viewer engagement ("Have you ever...")

4. **First 30 Seconds Analysis**: Specifically analyze how the video hooks viewers in the opening

Return a JSON array of detected hooks:
[
  {{
    "timestamp": 0.0,
    "end_timestamp": 5.0,
    "hook_type": "content",
    "description": "Opens with a provocative question to engage viewers",
    "confidence": 0.85,
    "transcript_snippet": "Have you ever wondered why..."
  }}
]

Guidelines:
- Focus on the most impactful hooks (quality over quantity)
- Confidence should reflect how likely this is to retain viewer attention
- Include transcript snippet when the hook involves speech
- timestamp is start of hook in seconds

Return ONLY the JSON array."""


def intro_analysis(transcript: str, frame_descriptions: str) -> str:
    """Generate intro analysis prompt."""
    return f"""Analyze this video's opening hook (first 30-60 seconds).

TRANSCRIPT:
{transcript}

FRAMES (in order):
{frame_descriptions}

Evaluate the intro's hook effectiveness:

1. **Hook Type**: What technique is used? (question, bold claim, story tease, visual surprise, etc.)
2. **Speed to Hook**: How quickly does the video grab attention?
3. **Promise**: What does the intro promise the viewer?
4. **Thumbnail-Title Alignment**: Based on the opening, does it likely match expectations set by title/thumbnail?

Return JSON:
{{
  "hook_technique": "description of the hook technique used",
  "hook_timestamp": 0.0,
  "speed_rating": "fast|medium|slow",
  "promise": "what the video promises to deliver",
  "effectiveness_score": 0.0-1.0,
  "improvement_suggestion": "how the hook could be stronger"
}}

Return ONLY the JSON object."""


# =============================================================================
# SEARCH RE-RANKING PROMPTS
# =============================================================================

def search_rerank(query: str, candidates: str) -> str:
    """Generate search re-ranking prompt."""
    return f"""Given the search query and candidate results, re-rank them by relevance.

QUERY: {query}

CANDIDATES:
{candidates}

For each candidate, assign a relevance score from 0.0 to 1.0 based on:
- Semantic relevance to the query (not just keyword matching)
- How well it answers what the user is looking for
- Context and intent behind the query

Return a JSON array of objects with "index" and "score":
[
  {{"index": 0, "score": 0.95}},
  {{"index": 1, "score": 0.72}},
  ...
]

Only include candidates with score >= 0.3. Order by score descending.
Return ONLY the JSON array."""
