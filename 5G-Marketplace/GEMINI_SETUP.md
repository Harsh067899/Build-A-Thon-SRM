# Gemini AI Integration Setup Guide

## 🤖 AI-Powered Slice Selection

This feature uses Google's Gemini AI (free tier) to analyze natural language descriptions and recommend the best 5G network slice type.

---

## 🔑 Step 1: Get Your Free Gemini API Key

### 1. Go to Google AI Studio
Visit: https://makersuite.google.com/app/apikey

### 2. Sign in with your Google account

### 3. Create API Key
- Click **"Create API Key"**
- Choose **"Create API key in new project"** (or select existing project)
- Copy the API key (it will look like: `AIzaSy...`)

⚠️ **Important:** Keep this key secret! Don't commit it to GitHub.

---

## 🚀 Step 2: Add API Key to Vercel

### Option A: Via Vercel Dashboard (Recommended)

1. Go to your Vercel project: https://vercel.com/dashboard
2. Select your `5g-marketplace` project
3. Go to **Settings** → **Environment Variables**
4. Add new variable:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** Your API key (paste it)
   - **Environments:** Check all (Production, Preview, Development)
5. Click **Save**

### Option B: Via Vercel CLI

```bash
cd /g/5g_sliced/5G-Marketplace
vercel env add GEMINI_API_KEY
# Paste your API key when prompted
# Select all environments (Production, Preview, Development)
```

---

## 📦 Step 3: Deploy Updated Code

### 1. Commit changes to GitHub

```bash
cd /g/5g_sliced
git add 5G-Marketplace/api/
git commit -m "Add Gemini AI integration for slice selection"
git push origin main
```

### 2. Deploy to Vercel

```bash
cd /g/5g_sliced/5G-Marketplace
vercel --prod
```

Or just push to GitHub and Vercel will auto-deploy (if you enabled it).

---

## ✅ Step 4: Test the Feature

### 1. Go to your deployed site
```
https://your-app.vercel.app
```

### 2. Try the AI-Powered Slice Selection box

**Example phrases to try:**
- "I need low latency for autonomous vehicles"
- "I need to connect thousands of IoT sensors"
- "I want to stream 4K video content"
- "I need reliable communication for industrial robots"
- "I have a smart city with millions of devices"

### 3. The AI will:
1. Analyze your requirement
2. Recommend a slice type (eMBB, URLLC, or mMTC)
3. Show confidence score
4. Redirect to vendor selection

---

## 🔧 How It Works

```
User Input → Gemini AI → Slice Type Analysis → Frontend Display
```

1. **User enters requirement** in plain English
2. **Gemini AI analyzes** using advanced language understanding
3. **System recommends** appropriate slice type:
   - **eMBB:** High bandwidth (video, AR/VR)
   - **URLLC:** Low latency (autonomous vehicles, surgery)
   - **mMTC:** Massive IoT (sensors, smart cities)
4. **User is redirected** to vendor selection

---

## 💰 Gemini API Free Tier Limits

✅ **Using `gemini-1.5-flash` - 100% FREE Forever!**

**Free Tier Limits (No Credit Card Required):**
- ✅ **15 requests per minute**
- ✅ **1,500 requests per day**
- ✅ **1 million tokens per month**
- ✅ **No billing required**
- ✅ **No credit card needed**

**Perfect for:**
- 🎓 Student projects
- 🏆 Hackathons
- 🧪 Demos & prototypes
- 📊 Small-scale applications

**This is MORE than enough for your 5G Marketplace demo!**

---

## 🎯 Model Used

We're using **`gemini-1.5-flash`** which is:
- ✅ Completely **FREE** (no payment required)
- ⚡ **Fast** response times
- 🎯 **Accurate** for classification tasks
- 💪 **Powerful** enough for complex reasoning

---

## 🐛 Troubleshooting

### Issue: "Gemini API is not configured"
**Solution:** Make sure you added the `GEMINI_API_KEY` environment variable in Vercel and redeployed.

### Issue: API returns error
**Solution:** 
1. Check your API key is valid at https://makersuite.google.com/app/apikey
2. Verify the key is correctly set in Vercel environment variables
3. Redeploy your app after adding the key

### Issue: Getting wrong slice recommendations
**Solution:** The AI learns from your prompts. Try being more specific:
- Instead of "fast internet" → "4K video streaming for 1000 users"
- Instead of "IoT" → "connecting 50,000 temperature sensors across a city"

---

## 📊 What Gets Analyzed

The AI looks for keywords and context:

**eMBB indicators:**
- video, streaming, broadband, AR, VR, high-definition
- bandwidth, throughput, capacity

**URLLC indicators:**
- latency, real-time, critical, autonomous, surgery
- industrial, automation, reliability

**mMTC indicators:**
- IoT, sensors, devices, massive, smart city
- monitoring, agriculture, environmental

---

## 🎉 You're Done!

Your 5G Marketplace now has AI-powered slice recommendations using Gemini!

Test it live and impress your users! 🚀
