"""
MPU6050 Signal Processor
Tremor detection using bandpass filtering and Power Spectral Density analysis
Based on teammate's tremor detection implementation
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone
from uuid import UUID
from scipy import signal

from .config import MPU6050Config

logger = logging.getLogger(__name__)


class MPU6050Processor:
    """
    Signal processing for MPU6050 data

    Post-session processing:
    - Bandpass filtering (4-6 Hz) to isolate tremor frequencies
    - Power Spectral Density (PSD) analysis using Welch's method
    - Tremor detection and classification (linear vs rotational)
    - Movement pattern analysis
    """

    def __init__(self, config: Optional[MPU6050Config] = None):
        """
        Initialize MPU6050 processor

        Args:
            config: MPU6050 configuration
        """
        self.config = config if config else MPU6050Config()

        logger.info("MPU6050 Processor initialized")
        logger.info(f"  Tremor detection: {self.config.tremor_freq_low}-{self.config.tremor_freq_high} Hz")
        logger.info(f"  Window size: {self.config.tremor_window_seconds}s ({self.config.window_size} samples)")

    def process_session_data(
            self,
            session_id: UUID,
            db_session,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> int:
        """
        Post-session processing: Tremor detection and movement analysis

        Args:
            session_id: Session UUID
            db_session: Database session
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Number of processed metric records created
        """
        try:
            from hero_core.database.models.sensors import SensorAccelerometer, SensorGyroscope, MetricsProcessed
        except ImportError:
            logger.error("Could not import database models")
            return 0

        logger.info(f"Starting post-session MPU6050 processing for session {session_id}")

        # Query accelerometer data
        accel_query = db_session.query(SensorAccelerometer).filter(
            SensorAccelerometer.session_id == session_id
        ).order_by(SensorAccelerometer.time)

        if start_time:
            accel_query = accel_query.filter(SensorAccelerometer.time >= start_time)
        if end_time:
            accel_query = accel_query.filter(SensorAccelerometer.time <= end_time)

        accel_samples = accel_query.all()

        # Query gyroscope data
        gyro_query = db_session.query(SensorGyroscope).filter(
            SensorGyroscope.session_id == session_id
        ).order_by(SensorGyroscope.time)

        if start_time:
            gyro_query = gyro_query.filter(SensorGyroscope.time >= start_time)
        if end_time:
            gyro_query = gyro_query.filter(SensorGyroscope.time <= end_time)

        gyro_samples = gyro_query.all()

        if len(accel_samples) == 0 or len(gyro_samples) == 0:
            logger.warning("No raw MPU6050 data found for session")
            return 0

        logger.info(f"Processing {len(accel_samples)} accel and {len(gyro_samples)} gyro samples")

        # Convert to numpy arrays
        accel_data = {
            'x': np.array([s.x for s in accel_samples]),
            'y': np.array([s.y for s in accel_samples]),
            'z': np.array([s.z for s in accel_samples]),
            'time': [s.time for s in accel_samples]
        }

        gyro_data = {
            'x': np.array([s.x for s in gyro_samples]),
            'y': np.array([s.y for s in gyro_samples]),
            'z': np.array([s.z for s in gyro_samples]),
            'time': [s.time for s in gyro_samples]
        }

        # Analyze tremor
        metrics_created = 0

        # Process in windows (use full session as one window if long enough)
        if len(accel_samples) >= self.config.window_size:
            tremor_results = self._analyze_tremor(accel_data, gyro_data)

            # Store tremor metrics
            metrics_created += self._store_tremor_metrics(
                session_id, db_session, tremor_results, accel_samples[-1].time
            )
        else:
            logger.warning(f"Insufficient data for tremor analysis: {len(accel_samples)} < {self.config.window_size}")

        # Commit all metrics
        db_session.commit()
        logger.info(f"✓ Post-session processing complete: {metrics_created} metrics created")

        return metrics_created

    def _analyze_tremor(
            self,
            accel_data: Dict[str, np.ndarray],
            gyro_data: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Analyze tremor using bandpass filtering and PSD analysis
        Exact implementation from teammate's tremor detection code

        Args:
            accel_data: Dictionary with x, y, z, time arrays
            gyro_data: Dictionary with x, y, z, time arrays

        Returns:
            Dictionary with tremor analysis results
        """
        # Calculate linear acceleration magnitude
        accel_magnitude = np.sqrt(
            accel_data['x'] ** 2 +
            accel_data['y'] ** 2 +
            accel_data['z'] ** 2
        )

        # Calculate rotational velocity magnitude
        gyro_magnitude = np.sqrt(
            gyro_data['x'] ** 2 +
            gyro_data['y'] ** 2 +
            gyro_data['z'] ** 2
        )

        # Band-pass filter to isolate tremor frequency (4-6 Hz)
        sos_accel = signal.butter(
            self.config.tremor_filter_order,
            [self.config.tremor_freq_low, self.config.tremor_freq_high],
            'bandpass',
            fs=self.config.sample_rate,
            output='sos'
        )
        filtered_accel = signal.sosfilt(sos_accel, accel_magnitude)

        sos_gyro = signal.butter(
            self.config.tremor_filter_order,
            [self.config.tremor_freq_low, self.config.tremor_freq_high],
            'bandpass',
            fs=self.config.sample_rate,
            output='sos'
        )
        filtered_gyro = signal.sosfilt(sos_gyro, gyro_magnitude)

        # Compute Power Spectral Density for accelerometer
        freqs_accel, psd_accel = signal.welch(
            filtered_accel,
            self.config.sample_rate,
            nperseg=self.config.sample_rate * 2
        )

        # Compute Power Spectral Density for gyroscope
        freqs_gyro, psd_gyro = signal.welch(
            filtered_gyro,
            self.config.sample_rate,
            nperseg=self.config.sample_rate * 2
        )

        # Find tremor in 4-6 Hz band (accelerometer)
        tremor_band_accel = (freqs_accel >= self.config.tremor_freq_low) & (freqs_accel <= self.config.tremor_freq_high)
        tremor_power_accel = np.sum(psd_accel[tremor_band_accel])
        total_power_accel = np.sum(psd_accel[freqs_accel <= 15])
        tremor_ratio_accel = tremor_power_accel / total_power_accel if total_power_accel > 0 else 0

        # Find tremor in 4-6 Hz band (gyroscope)
        tremor_band_gyro = (freqs_gyro >= self.config.tremor_freq_low) & (freqs_gyro <= self.config.tremor_freq_high)
        tremor_power_gyro = np.sum(psd_gyro[tremor_band_gyro])
        total_power_gyro = np.sum(psd_gyro[freqs_gyro <= 15])
        tremor_ratio_gyro = tremor_power_gyro / total_power_gyro if total_power_gyro > 0 else 0

        # Find peak frequencies
        peak_idx_accel = np.argmax(psd_accel[freqs_accel <= 10])
        peak_freq_accel = freqs_accel[freqs_accel <= 10][peak_idx_accel]

        peak_idx_gyro = np.argmax(psd_gyro[freqs_gyro <= 10])
        peak_freq_gyro = freqs_gyro[freqs_gyro <= 10][peak_idx_gyro]

        # Determine tremor status
        tremor_detected = False
        tremor_type = "none"

        if (tremor_ratio_accel > self.config.tremor_threshold or tremor_ratio_gyro > self.config.tremor_threshold) and \
                (self.config.tremor_freq_low <= peak_freq_accel <= self.config.tremor_freq_high or
                 self.config.tremor_freq_low <= peak_freq_gyro <= self.config.tremor_freq_high):
            tremor_detected = True

            if tremor_ratio_gyro > tremor_ratio_accel:
                tremor_type = "rotational"  # Pill-rolling pattern
            else:
                tremor_type = "linear"

        return {
            # Accelerometer results
            'accel_peak_freq': float(peak_freq_accel),
            'accel_tremor_ratio': float(tremor_ratio_accel),
            'accel_tremor_power': float(tremor_power_accel),

            # Gyroscope results
            'gyro_peak_freq': float(peak_freq_gyro),
            'gyro_tremor_ratio': float(tremor_ratio_gyro),
            'gyro_tremor_power': float(tremor_power_gyro),

            # Overall assessment
            'tremor_detected': tremor_detected,
            'tremor_type': tremor_type,
            'combined_ratio': float((tremor_ratio_accel + tremor_ratio_gyro) / 2)
        }

    def _store_tremor_metrics(
            self,
            session_id: UUID,
            db_session,
            tremor_results: Dict,
            timestamp: datetime
    ) -> int:
        """
        Store tremor detection metrics to database

        Args:
            session_id: Session UUID
            db_session: Database session
            tremor_results: Dictionary with tremor analysis results
            timestamp: Timestamp for metrics

        Returns:
            Number of metrics created
        """
        try:
            from hero_core.database.models.sensors import MetricsProcessed
        except ImportError:
            return 0

        metrics_created = 0

        # Store accelerometer tremor metrics
        metrics = [
            ('tremor_accel_peak_freq', tremor_results['accel_peak_freq']),
            ('tremor_accel_power_ratio', tremor_results['accel_tremor_ratio']),
            ('tremor_accel_power', tremor_results['accel_tremor_power']),

            # Store gyroscope tremor metrics
            ('tremor_gyro_peak_freq', tremor_results['gyro_peak_freq']),
            ('tremor_gyro_power_ratio', tremor_results['gyro_tremor_ratio']),
            ('tremor_gyro_power', tremor_results['gyro_tremor_power']),

            # Store overall tremor assessment
            ('tremor_detected', 1.0 if tremor_results['tremor_detected'] else 0.0),
            ('tremor_combined_ratio', tremor_results['combined_ratio']),
        ]

        for metric_type, value in metrics:
            metric = MetricsProcessed(
                time=timestamp,
                session_id=session_id,
                metric_type=metric_type,
                value=value,
                confidence=1.0,
                computation_method='tremor_detection_psd',
                is_valid=True
            )
            db_session.add(metric)
            metrics_created += 1

        # Store tremor type as a separate metric (encode as number)
        tremor_type_encoding = {
            'none': 0.0,
            'linear': 1.0,
            'rotational': 2.0
        }

        tremor_type_metric = MetricsProcessed(
            time=timestamp,
            session_id=session_id,
            metric_type='tremor_type',
            value=tremor_type_encoding.get(tremor_results['tremor_type'], 0.0),
            confidence=1.0,
            computation_method='tremor_detection_psd',
            is_valid=True
        )
        db_session.add(tremor_type_metric)
        metrics_created += 1

        # Log results
        logger.info("Tremor Analysis Results:")
        logger.info(f"  LINEAR (Accel): Peak={tremor_results['accel_peak_freq']:.2f}Hz, "
                    f"Ratio={tremor_results['accel_tremor_ratio']:.3f}")
        logger.info(f"  ROTATIONAL (Gyro): Peak={tremor_results['gyro_peak_freq']:.2f}Hz, "
                    f"Ratio={tremor_results['gyro_tremor_ratio']:.3f}")
        logger.info(f"  DETECTED: {tremor_results['tremor_detected']} "
                    f"({tremor_results['tremor_type']})")

        return metrics_created
