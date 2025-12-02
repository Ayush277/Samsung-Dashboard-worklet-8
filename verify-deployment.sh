#!/bin/bash

echo "🔍 Pre-Deployment Verification for Vercel"
echo "=========================================="
echo ""

# Check if vercel CLI is installed
echo "1️⃣ Checking Vercel CLI..."
if command -v vercel &> /dev/null; then
    echo "   ✅ Vercel CLI is installed"
    vercel --version
else
    echo "   ❌ Vercel CLI not found"
    echo "   📦 Install with: npm install -g vercel"
    exit 1
fi
echo ""

# Check Python version
echo "2️⃣ Checking Python version..."
python3 --version
echo ""

# Check file sizes
echo "3️⃣ Checking model file sizes (Vercel limit: 50MB per function)..."
echo ""
echo "Loan Risk Models:"
du -h "Loan delinquency risk/models/"*.pkl 2>/dev/null
echo ""
echo "Campaign Models:"
du -h "Campaign performance (marketing)/"*.pkl 2>/dev/null
echo ""
echo "Sales Models:"
du -h "Sell-out performance forecasting (sales uplift)/pipeline/"*.pkl 2>/dev/null
echo ""

# Check total deployment size
echo "4️⃣ Checking total deployment size..."
TOTAL_SIZE=$(du -sh . | cut -f1)
echo "   Total: $TOTAL_SIZE"
echo "   Note: Large CSV files are excluded via .vercelignore"
echo ""

# Verify required files exist
echo "5️⃣ Verifying deployment files..."
files=("vercel.json" "requirements.txt" ".vercelignore" "dashboard/app.py" "Loan delinquency risk/app.py" "Campaign performance (marketing)/app.py" "Sell-out performance forecasting (sales uplift)/pipeline/app.py")

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = true ]; then
    echo "✨ All required files are present!"
else
    echo "⚠️  Some required files are missing"
    exit 1
fi

echo ""
echo "6️⃣ Checking dependencies..."
if [ -f "requirements.txt" ]; then
    echo "   Dependencies in requirements.txt:"
    cat requirements.txt | grep -v "^#" | grep -v "^$"
fi
echo ""

echo "=========================================="
echo "✅ Pre-deployment checks complete!"
echo ""
echo "🚀 Ready to deploy! Run:"
echo "   vercel          (for preview deployment)"
echo "   vercel --prod   (for production deployment)"
echo ""
echo "📖 See VERCEL_DEPLOYMENT_GUIDE.md for detailed instructions"
