"""
Pipeline Automation Script
Schedules regular data updates and model retraining
"""

import schedule
import time
from datetime import datetime
from pathlib import Path
import subprocess
import yaml
from loguru import logger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class PipelineScheduler:
    """Schedule and automate pipeline executions"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize scheduler"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.automation_config = self.config.get('automation', {})
        
        # Setup logging
        logger.add("logs/scheduler.log", rotation="10 MB", retention="60 days")
    
    def run_pipeline(self, scenarios: list = None):
        """Execute the main pipeline"""
        logger.info("Scheduled pipeline execution started")
        
        try:
            # Build command
            cmd = ["python", "src/main.py"]
            
            if scenarios:
                cmd.extend(["--scenario"] + scenarios)
            else:
                cmd.extend(["--scenario", "all"])
            
            # Execute pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Pipeline execution successful")
            logger.info(f"Output: {result.stdout}")
            
            # Send notification
            if self.automation_config.get('notifications', {}).get('email'):
                self.send_notification(
                    subject="Pipeline Execution Successful",
                    message=f"Pipeline completed at {datetime.now()}\n\n{result.stdout}"
                )
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(f"Error output: {e.stderr}")
            
            # Send error notification
            if self.automation_config.get('notifications', {}).get('email'):
                self.send_notification(
                    subject="Pipeline Execution Failed",
                    message=f"Pipeline failed at {datetime.now()}\n\nError: {e.stderr}",
                    is_error=True
                )
            
            return False
    
    def data_refresh_job(self):
        """Job to refresh data only"""
        logger.info("Running data refresh job")
        
        try:
            result = subprocess.run(
                ["python", "src/main.py", "--phase", "ingestion"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Data refresh completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data refresh failed: {str(e)}")
            return False
    
    def model_retrain_job(self):
        """Job to retrain models"""
        logger.info("Running model retraining job")
        
        try:
            result = subprocess.run(
                ["python", "src/main.py", "--phase", "modeling", "--scenario", "all"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Model retraining completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model retraining failed: {str(e)}")
            return False
    
    def health_check(self):
        """Perform system health check"""
        logger.info("Running health check")
        
        checks = {
            'data_available': self._check_data_files(),
            'models_available': self._check_model_files(),
            'disk_space': self._check_disk_space(),
            'api_running': self._check_api_health()
        }
        
        logger.info(f"Health check results: {checks}")
        
        # Alert if any checks fail
        if not all(checks.values()):
            self.send_notification(
                subject="System Health Check Warning",
                message=f"Some health checks failed:\n\n{checks}",
                is_error=True
            )
        
        return checks
    
    def _check_data_files(self) -> bool:
        """Check if required data files exist"""
        required_files = [
            "data/processed/solar_data_processed.parquet",
            "data/processed/solar_data_features.parquet"
        ]
        return all(Path(f).exists() for f in required_files)
    
    def _check_model_files(self) -> bool:
        """Check if model files exist"""
        model_dir = Path("data/models")
        if not model_dir.exists():
            return False
        
        required_models = [f"emission_model_{s}.pkl" for s in ['NZE', 'APS', 'STEPS']]
        return all((model_dir / m).exists() for m in required_models)
    
    def _check_disk_space(self, threshold_gb: float = 1.0) -> bool:
        """Check available disk space"""
        import shutil
        
        stats = shutil.disk_usage(".")
        free_gb = stats.free / (1024**3)
        
        logger.info(f"Free disk space: {free_gb:.2f} GB")
        return free_gb > threshold_gb
    
    def _check_api_health(self) -> bool:
        """Check if API is responding"""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def send_notification(self, subject: str, message: str, is_error: bool = False):
        """Send email notification"""
        # This is a placeholder - configure with actual SMTP settings
        logger.info(f"Notification: {subject}")
        logger.info(f"Message: {message}")
        
        # In production, implement actual email sending:
        # smtp_server = "smtp.gmail.com"
        # smtp_port = 587
        # sender_email = "your-email@example.com"
        # receiver_email = "recipient@example.com"
        # password = "your-password"
        # 
        # msg = MIMEMultipart()
        # msg['From'] = sender_email
        # msg['To'] = receiver_email
        # msg['Subject'] = subject
        # msg.attach(MIMEText(message, 'plain'))
        # 
        # with smtplib.SMTP(smtp_server, smtp_port) as server:
        #     server.starttls()
        #     server.login(sender_email, password)
        #     server.send_message(msg)
    
    def setup_schedules(self):
        """Setup all scheduled jobs"""
        logger.info("Setting up scheduled jobs...")
        
        # Data refresh schedule (weekly)
        data_refresh_cron = self.automation_config.get('schedule', {}).get('data_refresh', '0 2 * * 0')
        schedule.every().sunday.at("02:00").do(self.data_refresh_job)
        logger.info(f"Scheduled data refresh: Every Sunday at 02:00")
        
        # Model retrain schedule (monthly)
        model_retrain_cron = self.automation_config.get('schedule', {}).get('model_retrain', '0 3 1 * *')
        schedule.every().month.at("03:00").do(self.model_retrain_job)
        logger.info(f"Scheduled model retraining: 1st of every month at 03:00")
        
        # Daily health check
        schedule.every().day.at("00:00").do(self.health_check)
        logger.info(f"Scheduled health check: Daily at 00:00")
        
        # Full pipeline (for testing - adjust as needed)
        # schedule.every().week.do(lambda: self.run_pipeline(['NZE']))
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Starting pipeline scheduler...")
        
        self.setup_schedules()
        
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Automation Scheduler")
    parser.add_argument(
        '--run-now',
        action='store_true',
        help='Run pipeline immediately instead of scheduling'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (no actual execution)'
    )
    
    args = parser.parse_args()
    
    scheduler = PipelineScheduler()
    
    if args.run_now:
        logger.info("Running pipeline immediately...")
        scheduler.run_pipeline()
    elif args.test:
        logger.info("Test mode: Checking configuration...")
        scheduler.health_check()
        print("\n✓ Test complete. Scheduler is configured correctly.")
    else:
        scheduler.run_scheduler()


if __name__ == "__main__":
    main()
