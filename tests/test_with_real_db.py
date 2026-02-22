#!/usr/bin/env python3
"""
Test both sensors with real database connection
"""

import sys
import time
import logging
from uuid import uuid4
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, '/home/hero/hero_system')
sys.path.insert(0, '/home/hero')
sys.path.insert(0, '/home/hero/hardware-calibration/tests/Accel_Oxi_TB/max30102-master')

def test_sensors():
    """Test both sensors with real database"""
    
    print("=" * 70)
    print("HERO System - Real Database Test")
    print("=" * 70)
    
    try:
        # Import modules
        from hero_system.coordinator.coordinator import SensorCoordinator
        from hero_system.sensors.mpu6050.collector import MPU6050Collector
        from hero_system.sensors.mpu6050.config import MPU6050Config
        from hero_system.sensors.max30102.collector import MAX30102Collector
        from hero_system.sensors.max30102.config import MAX30102Config
        
        print("\n✓ Modules imported")
        
        # Connect to database
        print("\n1. Connecting to database...")
        engine = create_engine('postgresql://postgres:pgdbadmin@localhost:5432/hero_db')
        Session = sessionmaker(bind=engine)
        db_session = Session()
        print("✓ Database connected")
        
        # Create session_id
        session_id = uuid4()
        user_id = uuid4()
        username = f"test_user_{int(time.time())}"  # Unique username
        print(f"✓ Session ID: {session_id}")
        
        # Create test user and session
        db_session.execute(
            text("INSERT INTO users (user_id, username) VALUES (:user_id, :username)"),
            {"user_id": user_id, "username": username}  # Change "test_user" to username
        )
        db_session.execute(
            text("INSERT INTO test_sessions (session_id, user_id) VALUES (:session_id, :user_id)"),
            {"session_id": session_id, "user_id": user_id}
        )
        db_session.commit()
        print("✓ Test session created")
        
        # Initialize coordinator
        print("\n2. Initializing coordinator...")
        coordinator = SensorCoordinator(session_id=session_id, db_session=db_session)
        print("✓ Coordinator ready")
        
        # Configure sensors
        mpu_config = MPU6050Config.for_session()
        max_config = MAX30102Config.for_session()
        
        # Initialize collectors
        print("\n3. Initializing sensors...")
        mpu_collector = MPU6050Collector(
            session_id=session_id,
            db_session=db_session,
            coordinator=coordinator,
            config=mpu_config
        )
        
        max_collector = MAX30102Collector(
            session_id=session_id,
            db_session=db_session,
            coordinator=coordinator,
            config=max_config
        )
        print("✓ Sensors initialized")
        
        # Start collectors
        print("\n4. Starting data collection...")
        print("-" * 70)
        mpu_collector.start()
        time.sleep(0.5)
        max_collector.start()
        
        print("\n✓ Both sensors running!")
        print("\nCollecting for 10 seconds...")
        print("Place finger on MAX30102 and move device\n")
        
        # Collect for 10 seconds
        start_time = time.time()
        last_status = 0
        
        while time.time() - start_time < 10:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_status >= 2.0:
                print(f"[{elapsed:.1f}s] MPU: {mpu_collector.accel_sample_count} accel, {mpu_collector.gyro_sample_count} gyro | MAX: {max_collector.sample_count}")
                last_status = current_time
            
            time.sleep(0.1)
        
        print("\n" + "=" * 70)
        print("Stopping...")
        max_collector.stop()
        mpu_collector.stop()
        
        print("\nFINAL COUNTS:")
        print(f"  MPU6050:  {mpu_collector.accel_sample_count} accel, {mpu_collector.gyro_sample_count} gyro")
        print(f"  MAX30102: {max_collector.sample_count}")
        
        # Verify in database
        print("\n5. Verifying database...")
        
        accel_count = db_session.execute(
            text("SELECT COUNT(*) FROM sensor_accelerometer WHERE session_id = :sid"),
            {"sid": session_id}
        ).scalar()
        
        gyro_count = db_session.execute(
            text("SELECT COUNT(*) FROM sensor_gyroscope WHERE session_id = :sid"),
            {"sid": session_id}
        ).scalar()
        
        hr_count = db_session.execute(
            text("SELECT COUNT(*) FROM sensor_heart_rate WHERE session_id = :sid"),
            {"sid": session_id}
        ).scalar()
        
        print(f"  DB: {accel_count} accel, {gyro_count} gyro, {hr_count} hr")
        print("=" * 70)
        print("\n✓ Test completed!\n")
        
        db_session.close()
        
    except KeyboardInterrupt:
        print("\n\nStopped early")
        try:
            max_collector.stop()
            mpu_collector.stop()
            db_session.close()
        except:
            pass
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    test_sensors()
    
