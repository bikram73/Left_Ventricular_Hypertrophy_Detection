"""
Fixed Data Download Script using Kaggle API directly
"""
import os
import zipfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_with_direct_api():
    """Download using Kaggle API directly without CLI"""
    try:
        # Import kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()
        
        # Create directories
        base_dir = Path(__file__).parent
        data_dir = base_dir / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        datasets = {
            'ecg': 'shayanfazeli/heartbeat',
            'mri': 'salikhussaini49/sunnybrook-cardiac-mri',
            'ct': 'nikhilroxtomar/ct-heart-segmentation', 
            'clinical': 'fedesoriano/heart-failure-prediction'
        }
        
        success_count = 0
        
        # Download each dataset
        for data_type, dataset_name in datasets.items():
            try:
                logger.info(f"Downloading {data_type} dataset: {dataset_name}")
                
                output_dir = data_dir / data_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Download and extract
                api.dataset_download_files(
                    dataset_name, 
                    path=str(output_dir), 
                    unzip=True
                )
                
                logger.info(f"✓ Successfully downloaded {data_type}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to download {data_type}: {str(e)}")
        
        # Organize data structure
        organize_data_structure(data_dir)
        
        if success_count == len(datasets):
            logger.info("🎉 All datasets downloaded successfully!")
            return True
        else:
            logger.warning(f"Downloaded {success_count}/{len(datasets)} datasets")
            return False
            
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        logger.error("Make sure kaggle.json is in C:\\Users\\Admin\\.kaggle\\kaggle.json")
        return False

def organize_data_structure(data_dir):
    """Organize downloaded data into proper structure"""
    # Create target subdirectories
    subdirs = {
        'ecg': 'heartbeat',
        'mri': 'sunnybrook', 
        'ct': 'ct_heart'
    }
    
    for data_type, subdir in subdirs.items():
        source_dir = data_dir / data_type
        target_dir = source_dir / subdir
        
        if source_dir.exists() and not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files to subdirectory
            for item in source_dir.iterdir():
                if item.is_file():
                    item.rename(target_dir / item.name)

def check_kaggle_config():
    """Check if Kaggle API is configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        logger.error(f"Kaggle API token not found at: {kaggle_json}")
        logger.error("Please:")
        logger.error("1. Go to https://www.kaggle.com/settings")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Save kaggle.json to: C:\\Users\\Admin\\.kaggle\\kaggle.json")
        return False
    
    logger.info("✓ Kaggle API token found")
    return True

def main():
    """Main download function"""
    print("=" * 60)
    print("🏥 LVH Detection - Dataset Download (Fixed Version)")
    print("=" * 60)
    
    # Check kaggle configuration
    if not check_kaggle_config():
        print("\n❌ Kaggle API not configured")
        print("Get API token from: https://www.kaggle.com/settings")
        return False
    
    # Try to download
    success = download_with_direct_api()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Next steps:")
        print("1. Run: python run.py")
        print("2. Access: http://localhost:5000")
        print("\nData structure created:")
        print("data/raw/")
        print("├── ecg/heartbeat/")
        print("├── mri/sunnybrook/")
        print("├── ct/ct_heart/")
        print("└── clinical/")
    else:
        print("\n" + "=" * 60)
        print("❌ DOWNLOAD FAILED")
        print("=" * 60)
        print("You can still run the app in demo mode:")
        print("python run.py")
    
    return success

if __name__ == "__main__":
    main()
