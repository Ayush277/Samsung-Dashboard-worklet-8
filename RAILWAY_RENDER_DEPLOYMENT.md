# 🚀 Samsung Dashboard - Railway & Render Deployment Guide

## ⚠️ Why Not Vercel?

Vercel's Python runtime **cannot install ML libraries** like XGBoost, LightGBM, and CatBoost. These require compilation which exceeds Vercel's build constraints.

**Solution:** Use Railway or Render instead - both fully support ML libraries!

---

## 🚂 Option 1: Railway (RECOMMENDED)

### Why Railway?
- ✅ Full ML library support
- ✅ No timeout limits
- ✅ More memory (up to 8GB)
- ✅ Persistent storage
- ✅ Easy deployment
- ✅ $5/month (includes $5 credit)

### Deployment Steps

#### 1. Install Railway CLI
```bash
npm install -g @railway/cli
```

#### 2. Login to Railway
```bash
railway login
```
This will open your browser to authenticate.

#### 3. Initialize Project
```bash
cd "/Users/ayush/Downloads/Samsung-Dashboard-worklet-8"
railway init
```

Follow the prompts:
- **Project name:** samsung-dashboard-worklet-8
- **Environment:** production

#### 4. Deploy!
```bash
railway up
```

That's it! Railway will:
- Install all dependencies (including ML libraries)
- Build your application
- Deploy to a public URL

#### 5. Get Your URL
```bash
railway domain
```

Or visit the Railway dashboard to see your deployed app.

### Expected URL
`https://samsung-dashboard-worklet-8-production.up.railway.app`

---

## 🎨 Option 2: Render (FREE TIER AVAILABLE)

### Why Render?
- ✅ Free tier available
- ✅ Full ML library support
- ✅ 15-minute timeout (great for ML)
- ✅ Easy GitHub integration
- ✅ Auto-deploy on push

### Deployment Steps

#### Method A: Using Render Dashboard (Easiest)

1. **Push to GitHub** (if not already)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Go to Render**
   - Visit https://render.com
   - Sign up / Log in
   - Click "New +" → "Web Service"

3. **Connect Repository**
   - Connect your GitHub account
   - Select your repository
   - Click "Connect"

4. **Configure Service**
   - **Name:** samsung-dashboard
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 dashboard.app:app`
   - **Plan:** Free

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

#### Method B: Using render.yaml (Automated)

1. **Push to GitHub** (render.yaml is already created)

2. **Go to Render Dashboard**
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will auto-detect `render.yaml`
   - Click "Apply"

### Expected URL
`https://samsung-dashboard.onrender.com`

---

## 📊 Comparison

| Feature | Railway | Render Free | Render Paid |
|---------|---------|-------------|-------------|
| **Cost** | $5/mo | Free | $7/mo |
| **Memory** | 8GB | 512MB | 2GB+ |
| **Timeout** | Unlimited | 15 min | 15 min |
| **Sleep** | No | Yes (15min) | No |
| **Build Time** | Fast | Slower | Fast |
| **Best For** | Production | Testing | Production |

---

## 🎯 My Recommendation

### For Testing
**Use Render Free Tier**
- No cost
- Good for demos
- Sleeps after 15min inactivity (wakes on request)

### For Production
**Use Railway**
- Better performance
- No sleep
- More memory
- Only $5/month

---

## 📝 Files Created for Deployment

✅ `requirements.txt` - All dependencies including ML libraries
✅ `Procfile` - Start command for Railway/Render
✅ `render.yaml` - Render configuration
✅ `.vercelignore` - Exclude large files (works for all platforms)

---

## 🚀 Quick Start Commands

### Railway
```bash
npm install -g @railway/cli
railway login
railway init
railway up
railway domain
```

### Render
Just push to GitHub and use the dashboard!

---

## ⚡ After Deployment

Your apps will be available at:
- **Dashboard:** `https://your-app-url.com/`
- **Loan Risk:** `https://your-app-url.com/loan/`
- **Campaign:** `https://your-app-url.com/campaign/`
- **Sales:** `https://your-app-url.com/sales/`

---

## 🐛 Troubleshooting

### Railway Issues
```bash
# View logs
railway logs

# Check status
railway status

# Redeploy
railway up --detach
```

### Render Issues
- Check logs in Render dashboard
- Verify build command is correct
- Check environment variables

---

## 💡 Tips

1. **First deployment takes longer** (10-15 minutes) due to ML library compilation
2. **Subsequent deployments are faster** (cached dependencies)
3. **Monitor memory usage** in the dashboard
4. **Set up custom domain** in platform settings

---

## 🎉 Ready to Deploy!

**Recommended:** Start with Render Free Tier to test, then upgrade to Railway for production.

Choose your platform and follow the steps above!

Need help? Check the platform documentation:
- Railway: https://docs.railway.app
- Render: https://render.com/docs
