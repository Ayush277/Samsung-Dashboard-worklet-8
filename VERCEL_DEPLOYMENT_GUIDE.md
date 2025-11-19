# 🚀 Vercel Deployment Guide for Samsung Dashboard Worklet 8

## 📋 Pre-Deployment Checklist

### 1. **Repository Structure**
Your project is now configured for Vercel deployment with:
- `vercel.json` - Vercel configuration
- `api/index.py` - Unified Flask application
- `requirements.txt` - Consolidated dependencies

### 2. **File Structure for Vercel**
```
Samsung-Dashboard-worklet-8/
├── vercel.json                 # Vercel config
├── requirements.txt            # Python dependencies
├── api/
│   └── index.py               # Main Flask app
├── dashboard/templates/        # HTML templates
├── Loan delinquency risk/     # Loan app logic
├── Campaign performance/      # Campaign app logic
└── Sell-out performance/      # Sales app logic
```

## 🔧 Deployment Steps

### Option 1: Deploy via Vercel CLI

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy from your project directory**
```bash
cd "/Users/ayush/Samsung-Dashboard-worklet-8"
vercel
```

4. **Follow the prompts:**
   - Set up and deploy? **Y**
   - Which scope? Choose your account
   - Link to existing project? **N** (for first deployment)
   - Project name: `samsung-dashboard-worklet-8`
   - Directory to deploy: `.` (current directory)

### Option 2: Deploy via GitHub Integration

1. **Push to GitHub**
```bash
git add .
git commit -m "Configure for Vercel deployment"
git push origin main
```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

## ⚙️ Important Configuration Notes

### 1. **Environment Variables**
If your app needs environment variables, add them in Vercel dashboard:
- Go to your project settings
- Navigate to "Environment Variables"
- Add any required variables

### 2. **File Size Limits**
- Vercel has a 250MB deployment limit
- Large ML models may need to be loaded from external sources
- Consider using Hugging Face Hub or other model repositories

### 3. **Cold Start Considerations**
- First request may be slow due to model loading
- Consider implementing model caching strategies
- Use Vercel's edge functions for better performance

## 🔍 Troubleshooting Common Issues

### Issue 1: Module Import Errors
If you get import errors, ensure all required modules are in `requirements.txt`

### Issue 2: Template Not Found
Templates should be in `dashboard/templates/` directory

### Issue 3: Model Loading Timeout
- Reduce model complexity for initial deployment
- Implement lazy loading for heavy models
- Use serverless-friendly model formats

### Issue 4: Memory Limits
Vercel free tier has memory limitations:
- Hobby: 1024MB
- Pro: 1024MB (can be increased)

## 📝 Post-Deployment Testing

1. **Test the main dashboard**: `https://your-app.vercel.app/`
2. **Test individual apps**:
   - Loan Risk: `https://your-app.vercel.app/loan`
   - Campaign: `https://your-app.vercel.app/campaign`
   - Sales: `https://your-app.vercel.app/sales`

## 🎯 Key Changes Made for Vercel Compatibility

1. **Unified Flask App**: Combined all separate Flask apps into one serverless function
2. **Removed Subprocess Management**: No more port management or subprocess spawning
3. **Serverless Architecture**: Each route is now a serverless function call
4. **Template Path Updates**: Consolidated template access
5. **Dependency Management**: Single requirements.txt file

## 📞 Support

If you encounter issues:
1. Check Vercel deployment logs
2. Review function logs in Vercel dashboard
3. Test locally first: `python api/index.py`

---

**Ready to deploy!** 🚀

Use `vercel` command in your project directory to get started.