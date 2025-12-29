"""
Main entry point for LVH Detection System
Run this file to start the Flask web application
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from app import app
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in the same directory")
    sys.exit(1)

def main():
    """Main function to start the application"""
    try:
        # Ensure upload directory exists
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("🏥 LVH Detection System Starting...")
        print("=" * 60)
        print(f"📍 Access the application at: http://localhost:5000")
        print(f"📊 Upload data at: http://localhost:5000/upload")
        print(f"🔍 API documentation at: http://localhost:5000/api")
        print(f"❤️  System health check: http://localhost:5000/health")
        print("=" * 60)
        print("🎯 Best Model Accuracies:")
        print("   • Clinical: 89.13% (GradientBoosting)")
        print("   • ECG:      82.00% (XGBoost)")
        print("   • MRI:      81.43% (SVM)")
        print("   • CT:       78.75% (LogisticRegression)")
        print("=" * 60)
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
        )
    except KeyboardInterrupt:
        print("\n👋 LVH Detection System stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)