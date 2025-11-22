#!/bin/bash
# Deploy script for Google Colab
# Syncs example.py to example.ipynb and pushes to GitHub

set -e  # Exit on error

echo "🚀 Starting Colab deployment process..."
echo ""

# Step 1: Check if jupytext is available
echo "📋 Step 1: Checking jupytext..."
if ! command -v jupytext &> /dev/null; then
    echo "❌ jupytext not found. Please install it first:"
    echo "   pip install jupytext"
    exit 1
fi
echo "✅ jupytext is available"
echo ""

# Step 2: Sync example.py to example.ipynb
echo "📋 Step 2: Syncing example.py → example.ipynb..."
jupytext --sync example.py
echo "✅ Notebook synced successfully"
echo ""

# Step 3: Verify the notebook was created/updated
if [ -f "example.ipynb" ]; then
    echo "✅ example.ipynb exists and is ready"
    ls -lh example.ipynb
else
    echo "❌ example.ipynb was not created"
    exit 1
fi
echo ""

# Step 4: Git add
echo "📋 Step 4: Adding files to git..."
git add example.py example.ipynb
echo "✅ Files staged"
echo ""

# Step 5: Commit and push
echo "📋 Step 5: Committing and pushing..."
git commit -m "Update example notebook for Colab deployment"

git push
echo "✅ Pushed to GitHub"
echo ""

# Step 6: Generate Colab link
echo "🔗 Colab Link:"
echo "https://colab.research.google.com/github/greenfieldcapital-labs/mci/blob/release-v1/example.ipynb"
echo ""

echo "✅ Deployment complete! Test in Colab using the link above."
