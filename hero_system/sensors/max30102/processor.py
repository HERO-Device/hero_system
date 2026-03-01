"""
MAX30102 Signal Processor
HR, SpO2, and HRV computation from raw IR/Red signals
"""

import logging
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timezone
from uuid import UUID
from collections import deque

import hrcalc

from .config import MAX30102Config

logger = logging.getLogger(__name__)


class MAX30102Processor:
    """
    Signal processing for MAX30102 data

    Two modes:
    - Real-time: For calibration UI display (calculates but doesn't store)
    - Post-session: Queries raw data from DB, processes, stores to MetricsProcessed
    """

    def __init__(self, config: Optional[MAX30102Config] = None):
        """
        Initialize MAX30102 processor

        Args:
            config: MAX30102 configuration
        """
        self.config = config if config else MAX30102Config()

        # Buffers for real-time processing
        self.ir_buffer = deque(maxlen=self.config.buffer_size)
        self.red_buffer = deque(maxlen=self.config.buffer_size)
        self.hr_smoothing_buffer = deque(maxlen=self.config.hr_smoothing_window)

        logger.info("MAX30102 Processor initialized")

    def process_realtime(self, ir: int, red: int) -> Tuple[Optional[int], bool, Optional[int], bool]:
        """
        Process raw signals for real-time display (calibration mode)
        Does NOT store to database - just returns values

        Args:
            ir: Infrared signal value
            red: Red signal value

        Returns:
            Tuple of (hr, hr_valid, spo2, spo2_valid)
        """
        # Add to buffers
        self.ir_buffer.append(ir)
        self.red_buffer.append(red)

        # Need full buffer for calculation
        if len(self.ir_buffer) < self.config.buffer_size:
            return None, False, None, False

        # Convert to lists for hrcalc
        ir_data = list(self.ir_buffer)
        red_data = list(self.red_buffer)

        try:
            # Calculate HR and SpO2
            hr, hr_valid, spo2, spo2_valid = hrcalc.calc_hr_and_spo2(ir_data, red_data)

            # Smooth HR if valid
            if hr_valid:
                self.hr_smoothing_buffer.append(hr)
                if len(self.hr_smoothing_buffer) > 0:
                    hr = int(np.mean(self.hr_smoothing_buffer))
                else:
                    hr = int(hr)

            if spo2_valid:
                spo2 = int(spo2)

            return hr, hr_valid, spo2, spo2_valid

        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            return None, False, None, False

    def reset_buffers(self):
        """
        Clear all internal signal and smoothing buffers.

        Should be called when starting a new measurement to avoid
        stale data affecting calculations.

        Returns:
            None.
        """
        self.ir_buffer.clear()
        self.red_buffer.clear()
        self.hr_smoothing_buffer.clear()

    def process_session_data(
            self,
            session_id: UUID,
            db_session,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> int:
        """
        Post-session processing: Query raw data and compute HR/SpO2 metrics

        Args:
            session_id: Session UUID
            db_session: Database session
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Number of processed metric records created
        """
        try:
            from hero_core.database.models.sensors import SensorHeartRate, SensorOximeter, MetricsProcessed
        except ImportError:
            logger.error("Could not import database models")
            return 0

        logger.info(f"Starting post-session MAX30102 processing for session {session_id}")

        # Query raw IR signals (from SensorHeartRate)
        hr_query = db_session.query(SensorHeartRate).filter(
            SensorHeartRate.session_id == session_id
        ).order_by(SensorHeartRate.time)

        if start_time:
            hr_query = hr_query.filter(SensorHeartRate.time >= start_time)
        if end_time:
            hr_query = hr_query.filter(SensorHeartRate.time <= end_time)

        hr_samples = hr_query.all()

        # Query raw Red signals (from SensorOximeter)
        ox_query = db_session.query(SensorOximeter).filter(
            SensorOximeter.session_id == session_id
        ).order_by(SensorOximeter.time)

        if start_time:
            ox_query = ox_query.filter(SensorOximeter.time >= start_time)
        if end_time:
            ox_query = ox_query.filter(SensorOximeter.time <= end_time)

        ox_samples = ox_query.all()

        if len(hr_samples) == 0 or len(ox_samples) == 0:
            logger.warning("No raw MAX30102 data found for session")
            return 0

        logger.info(f"Processing {len(hr_samples)} IR samples and {len(ox_samples)} Red samples")

        # Process in sliding windows
        metrics_created = 0
        window_size = self.config.buffer_size

        for i in range(0, len(hr_samples), window_size // 2):  # 50% overlap
            # Get window of samples
            hr_window = hr_samples[i:i + window_size]
            ox_window = ox_samples[i:i + window_size]

            if len(hr_window) < window_size or len(ox_window) < window_size:
                logger.debug(f"Skipping small window: {len(hr_window)} samples")
                continue

            # Extract IR and Red signals
            ir_data = [float(s.raw_signal) for s in hr_window]
            red_data = [float(s.red_signal) for s in ox_window]

            try:
                # Calculate HR and SpO2
                hr, hr_valid, spo2, spo2_valid = hrcalc.calc_hr_and_spo2(ir_data, red_data)

                # Use timestamp of middle sample
                timestamp = hr_window[len(hr_window) // 2].time

                # Store HR metric
                if hr_valid:
                    hr_metric = MetricsProcessed(
                        time=timestamp,
                        session_id=session_id,
                        metric_type='heart_rate_bpm',
                        value=float(hr),
                        confidence=1.0,
                        computation_method='hrcalc_postsession',
                        is_valid=True
                    )
                    db_session.add(hr_metric)
                    metrics_created += 1

                # Store SpO2 metric
                if spo2_valid:
                    spo2_metric = MetricsProcessed(
                        time=timestamp,
                        session_id=session_id,
                        metric_type='spo2_percent',
                        value=float(spo2),
                        confidence=1.0,
                        computation_method='hrcalc_postsession',
                        is_valid=True
                    )
                    db_session.add(spo2_metric)
                    metrics_created += 1

                # Commit periodically
                if metrics_created % 20 == 0:
                    db_session.commit()
                    logger.debug(f"Processed {i + len(hr_window)}/{len(hr_samples)} samples")

            except Exception as e:
                logger.error(f"Error processing window at index {i}: {e}")
                continue

        # Calculate HRV if we have enough HR data
        hrv_metrics = self._calculate_hrv(session_id, db_session)
        metrics_created += hrv_metrics

        # Final commit
        db_session.commit()
        logger.info(f"✓ Post-session processing complete: {metrics_created} metrics created")

        return metrics_created

    def _calculate_hrv(self, session_id: UUID, db_session) -> int:
        """
        Calculate Heart Rate Variability metrics

        Args:
            session_id: Session UUID
            db_session: Database session

        Returns:
            Number of HRV metrics created
        """
        try:
            from hero_core.database.models.sensors import MetricsProcessed
        except ImportError:
            return 0

        try:
            # Query all HR measurements for this session
            hr_metrics = db_session.query(MetricsProcessed).filter(
                MetricsProcessed.session_id == session_id,
                MetricsProcessed.metric_type == 'heart_rate_bpm',
                MetricsProcessed.is_valid == True
            ).order_by(MetricsProcessed.time).all()

            if len(hr_metrics) < 10:
                logger.debug("Not enough HR data for HRV calculation")
                return 0

            # Extract HR values
            hr_values = [m.value for m in hr_metrics]

            # Calculate inter-beat intervals (RR intervals) in milliseconds
            # RR interval = 60000 / HR (bpm)
            rr_intervals = [60000.0 / hr for hr in hr_values if hr > 0]

            if len(rr_intervals) < 10:
                return 0

            # Calculate HRV metrics
            # SDNN: Standard deviation of RR intervals
            sdnn = float(np.std(rr_intervals))

            # RMSSD: Root mean square of successive differences
            successive_diffs = np.diff(rr_intervals)
            rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

            # Use timestamp of last HR measurement
            timestamp = hr_metrics[-1].time

            # Store SDNN metric
            sdnn_metric = MetricsProcessed(
                time=timestamp,
                session_id=session_id,
                metric_type='hrv_sdnn',
                value=sdnn,
                confidence=1.0,
                computation_method='hrv_time_domain',
                is_valid=True
            )
            db_session.add(sdnn_metric)

            # Store RMSSD metric
            rmssd_metric = MetricsProcessed(
                time=timestamp,
                session_id=session_id,
                metric_type='hrv_rmssd',
                value=rmssd,
                confidence=1.0,
                computation_method='hrv_time_domain',
                is_valid=True
            )
            db_session.add(rmssd_metric)

            logger.info(f"✓ HRV calculated: SDNN={sdnn:.2f}ms, RMSSD={rmssd:.2f}ms")

            return 2  # Created 2 metrics

        except Exception as e:
            logger.error(f"Error calculating HRV: {e}")
            return 0