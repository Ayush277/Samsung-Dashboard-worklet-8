# Vercel Deployment Guide for Samsung Dashboard Worklet 8

## 📋 Overview
This guide will help you deploy the Samsung Dashboard with all three AI applications to Vercel.

## ⚠️ Important Notes

### Vercel Limitations
- **File Size**: Each serverless function has a 50MB limit (our models are within this)
- **Execution Time**: Free tier has 10s timeout, Pro has 60s (configured in vercel.json)
- **Memory**: Free tier has 1GB RAM, Pro has up to 3GB (we've configured 3GB for ML apps)
- **File System**: Read-only except for `/tmp` directory

### What's Been Configured
1. ✅ All apps updated to use absolute paths
2. ✅ Upload folders configured to use `/tmp` on Vercel
3. ✅ Separate serverless functions for each app
4. ✅ Proper routing configured
5. ✅ Memory limits increased for ML workloads

## 🚀 Deployment Steps

### Step 1: Install Vercel CLI (if not already installed)
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy from Project Directory
```bash
cd "/Users/ayush/Downloads/Samsung-Dashboard-worklet-8"
vercel
```

Follow the prompts:
- **Set up and deploy?** Yes
- **Which scope?** Select your account
- **Link to existing project?** No (first time) or Yes (subsequent deploys)
- **Project name?** samsung-dashboard-worklet-8 (or your preferred name)
- **Directory?** ./ (current directory)
- **Override settings?** No

### Step 4: Production Deployment
```bash
vercel --prod
```

## 🔧 Configuration Files Created

### `vercel.json`
- Routes each app to its own serverless function
- Configures memory (3GB) and timeout (60s) for ML apps
- Sets up proper URL routing:
  - `/` → Dashboard
  - `/loan` → Loan Risk Assessment
  - `/campaign` → Campaign Performance
  - `/sales` → Sales Forecasting

### `requirements.txt`
All necessary Python dependencies for the ML models.

### `.vercelignore`
Excludes large CSV training files and temporary directories.

## 📊 Expected URLs After Deployment

After deployment, you'll get a URL like: `https://samsung-dashboard-worklet-8.vercel.app`

- **Dashboard**: `https://your-app.vercel.app/`
- **Loan Risk**: `https://your-app.vercel.app/loan/`
- **Campaign**: `https://your-app.vercel.app/campaign/`
- **Sales**: `https://your-app.vercel.app/sales/`

## ⚡ Performance Considerations

### Cold Starts
- First request may take 5-10 seconds (loading ML models)
- Subsequent requests will be faster
- Consider Vercel Pro for better performance

### File Uploads
- Batch processing works but limited by execution time
- Large CSV files may timeout on free tier
- Consider upgrading to Pro for 60s timeout

### Model Loading
- Models are loaded on each cold start
- Keep models optimized and compressed
- Consider model caching strategies

## 🐛 Troubleshooting

### Deployment Fails
1. Check file sizes: `du -sh */*.pkl`
2. Verify all paths are absolute (not relative)
3. Check Vercel build logs: `vercel logs`

### Runtime Errors
1. Check function logs in Vercel dashboard
2. Verify environment variables
3. Test locally first: `vercel dev`

### Timeout Issues
1. Upgrade to Vercel Pro for 60s timeout
2. Optimize model loading
3. Use smaller batch sizes

## 💡 Recommendations

### For Production Use
1. **Upgrade to Vercel Pro** for:
   - Longer execution time (60s vs 10s)
   - More memory (3GB vs 1GB)
   - Better performance
   - No cold starts on popular routes

2. **Optimize Models**:
   - Consider model quantization
   - Use lighter model variants
   - Implement model caching

3. **Add Monitoring**:
   - Set up Vercel Analytics
   - Monitor function execution times
   - Track error rates

### Alternative: Docker Deployment
If Vercel's serverless limitations are too restrictive, consider:
- **Railway**: Better for ML apps, supports Docker
- **Render**: Good Python support, longer timeouts
- **Google Cloud Run**: Serverless containers, more flexible
- **AWS Lambda**: With container support

## 📝 Local Testing Before Deploy

Test the Vercel environment locally:
```bash
vercel dev
```

This will simulate the Vercel environment on your machine.

## 🔄 Continuous Deployment

Connect your GitHub repository to Vercel for automatic deployments:
1. Push code to GitHub
2. Import project in Vercel dashboard
3. Every push to main branch auto-deploys

## 📞 Support

If you encounter issues:
- Check Vercel documentation: https://vercel.com/docs
- Review deployment logs in Vercel dashboard
- Test locally with `vercel dev` first

---

**Ready to deploy?** Run `vercel` in the project directory!
