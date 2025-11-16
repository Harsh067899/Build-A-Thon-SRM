from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
import json
import os
import re
import google.generativeai as genai

app = FastAPI(title="5G Marketplace API - Vercel")

# Configure Gemini API (Using FREE tier model - gemini-2.5-flash)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC9OBTt36JehrYA6UYQMpBc6PUbIEP7JF0")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-2.5-flash - Completely FREE with 1500 requests/day
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load static data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@app.get("/")
def read_root():
    return {"message": "5G Marketplace API", "status": "running"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "platform": "vercel"}

@app.get("/api/dashboard/system-health")
def system_health():
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "database": "operational",
            "ai_agent": "disabled_on_vercel"
        },
        "metrics": {
            "total_slices": 0,
            "active_slices": 0,
            "total_vendors": 5
        }
    }

@app.get("/api/vendors")
def get_vendors():
    try:
        vendors_file = os.path.join(DATA_DIR, "vendors.json")
        with open(vendors_file, 'r') as f:
            vendors = json.load(f)
        return {"vendors": vendors}
    except:
        return {"vendors": []}

@app.get("/api/slices")
def get_slices():
    try:
        slices_file = os.path.join(DATA_DIR, "slices.json")
        with open(slices_file, 'r') as f:
            slices = json.load(f)
        return {"slices": slices}
    except:
        return {"slices": []}

# Request models
class SliceTypeRequest(BaseModel):
    text: str

# AI-Powered Slice Type Prediction using Gemini
@app.post("/api/slice-selection/predict-slice-type")
async def predict_slice_type(request: SliceTypeRequest):
    """Predict slice type from natural language using Gemini AI"""
    
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API is not configured")
    
    try:
        # Create an optimized prompt for Gemini Flash
        prompt = f"""You are a 5G network slicing expert. Analyze this user requirement and classify it into ONE of these exact slice types:

**Slice Types:**
1. **eMBB** (Enhanced Mobile Broadband)
   - Use cases: Video streaming, AR/VR, cloud gaming, high-definition content delivery, large file transfers
   - Key indicators: bandwidth, throughput, high-speed, streaming, video, download, upload, capacity

2. **URLLC** (Ultra-Reliable Low-Latency Communications)
   - Use cases: Autonomous vehicles, industrial automation, remote surgery, robotics, critical infrastructure
   - Key indicators: latency, real-time, critical, reliable, instant, autonomous, surgery, industrial, millisecond, safety

3. **mMTC** (Massive Machine Type Communications)
   - Use cases: IoT sensors, smart cities, smart agriculture, environmental monitoring, massive device connectivity
   - Key indicators: IoT, sensors, devices, monitoring, smart city, agriculture, massive, thousands, millions, low-power

**User Requirement:**
"{request.text}"

**Task:** Analyze the requirement and respond with ONLY valid JSON (no markdown, no explanations outside JSON):

{{"slice_type": "eMBB", "confidence": 0.95, "reasoning": "User needs high bandwidth for video streaming"}}

Rules:
- slice_type must be exactly: eMBB, URLLC, or mMTC
- confidence must be between 0.0 and 1.0
- reasoning should be 1-2 sentences explaining why"""

        # Call Gemini API with safety settings for consistent JSON output
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more consistent output
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 200,
        }
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            # Validate the response
            if "slice_type" not in result:
                raise ValueError("Missing slice_type in response")
            
            # Normalize slice_type to uppercase
            result["slice_type"] = result["slice_type"].upper()
            
            # Ensure valid slice type
            valid_types = ["EMBB", "URLLC", "MMTC"]
            if result["slice_type"] not in valid_types:
                # Try to map common variations
                slice_type_lower = result["slice_type"].lower()
                if "embb" in slice_type_lower or "broadband" in slice_type_lower:
                    result["slice_type"] = "eMBB"
                elif "urllc" in slice_type_lower or "latency" in slice_type_lower:
                    result["slice_type"] = "URLLC"
                elif "mmtc" in slice_type_lower or "iot" in slice_type_lower or "machine" in slice_type_lower:
                    result["slice_type"] = "mMTC"
                else:
                    raise ValueError(f"Invalid slice type: {result['slice_type']}")
            
            # Ensure confidence is present and valid
            if "confidence" not in result:
                result["confidence"] = 0.85
            else:
                result["confidence"] = float(result["confidence"])
                if result["confidence"] < 0 or result["confidence"] > 1:
                    result["confidence"] = 0.85
            
            return result
            
        except json.JSONDecodeError:
            # Fallback: try to extract slice type from text
            response_lower = response_text.lower()
            
            if "embb" in response_lower or "broadband" in response_lower:
                slice_type = "eMBB"
                confidence = 0.8
            elif "urllc" in response_lower or "latency" in response_lower:
                slice_type = "URLLC"
                confidence = 0.8
            elif "mmtc" in response_lower or "iot" in response_lower or "machine" in response_lower:
                slice_type = "mMTC"
                confidence = 0.8
            else:
                # Default to eMBB if uncertain
                slice_type = "eMBB"
                confidence = 0.6
            
            return {
                "slice_type": slice_type,
                "confidence": confidence,
                "reasoning": response_text[:200]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Vercel serverless handler
handler = Mangum(app)
