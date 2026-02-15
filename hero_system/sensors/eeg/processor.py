"""
EEG Signal Processing Utilities
Filtering, artifact removal, and band power computation
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from uuid import UUID

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from sqlalchemy.orm import Session

from .config import EEGConfig

logger = logging.getLogger(__name__)


class EEGProcessor:
    """
    Signal processing utilities for EEG data
    Handles filtering, artifact removal, and band power computation
    """

    def __init__(self, config: EEGConfig, sampling_rate: int):
        """
        Initialize EEG processor

        Args:
            config: EEG configuration
            sampling_rate: Sampling rate in Hz
        """
        self.config = config
        self.sampling_rate = sampling_rate
        self.psd_size = DataFilter.get_nearest_power_of_two(sampling_rate)

        logger.info(f"EEG Processor initialized: {sampling_rate} Hz, PSD size: {self.psd_size}")

    def apply_filters(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Apply standard EEG filters to channel data

        Args:
            channel_data: Raw EEG data for single channel

        Returns:
            Filtered EEG data
        """
        # Create a copy to avoid modifying original
        filtered_data = channel_data.copy()

        # Remove DC offset (detrend)
        DataFilter.detrend(filtered_data, DetrendOperations.CONSTANT.value)

        # Bandpass filter (3-45 Hz)
        DataFilter.perform_bandpass(
            filtered_data,
            self.sampling_rate,
            self.config.bandpass_low,
            self.config.bandpass_high,
            self.config.bandpass_order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0
        )

        # Notch filter for 50 Hz power line interference
        DataFilter.perform_bandstop(
            filtered_data,
            self.sampling_rate,
            self.config.notch_50hz[0],
            self.config.notch_50hz[1],
            self.config.notch_order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0
        )

        # Notch filter for 60 Hz power line interference
        DataFilter.perform_bandstop(
            filtered_data,
            self.sampling_rate,
            self.config.notch_60hz[0],
            self.config.notch_60hz[1],
            self.config.notch_order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0
        )

        return filtered_data

    def compute_band_powers(self, channel_data: np.ndarray) -> Dict[str, float]:
        """
        Compute power in each frequency band for a single channel

        Args:
            channel_data: Filtered EEG data for single channel

        Returns:
            Dictionary mapping band names to power values
        """
        if len(channel_data) < self.psd_size:
            logger.warning(f"Insufficient data for PSD: {len(channel_data)} < {self.psd_size}")
            return {
                'delta': 0.0,
                'theta': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'gamma': 0.0,
            }

        # Compute power spectral density using Welch's method
        psd_data = DataFilter.get_psd_welch(
            channel_data,
            self.psd_size,
            self.psd_size // 2,
            self.sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value
        )

        # Extract band powers
        bands = self.config.bands.to_dict()
        band_powers = {}

        for band_name, (low_freq, high_freq) in bands.items():
            power = DataFilter.get_band_power(psd_data, low_freq, high_freq)
            band_powers[band_name] = float(power)

        return band_powers

    def compute_multi_channel_band_powers(
        self,
        data: np.ndarray,
        eeg_channels: List[int]
    ) -> Dict[str, float]:
        """
        Compute average band powers across multiple EEG channels

        Args:
            data: Board data array (channels x samples)
            eeg_channels: List of EEG channel indices

        Returns:
            Dictionary with average band powers across channels
        """
        # Initialize accumulators
        avg_bands = {
            'delta': 0.0,
            'theta': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
        }

        valid_channels = 0

        # Process each channel (limit to num_channels from config)
        for channel_idx in eeg_channels[:self.config.num_channels]:
            try:
                # Extract and filter channel data
                channel_data = data[channel_idx].copy()
                filtered_data = self.apply_filters(channel_data)

                # Compute band powers
                band_powers = self.compute_band_powers(filtered_data)

                # Accumulate
                for band_name, power in band_powers.items():
                    avg_bands[band_name] += power

                valid_channels += 1

            except Exception as e:
                logger.error(f"Error processing channel {channel_idx}: {e}")
                continue

        # Average across channels
        if valid_channels > 0:
            for band_name in avg_bands:
                avg_bands[band_name] /= valid_channels
        else:
            logger.warning("No valid channels processed")

        return avg_bands

    def assess_signal_quality(self, channel_data: np.ndarray) -> Tuple[float, bool]:
        """
        Assess signal quality for a channel

        Args:
            channel_data: Raw EEG data

        Returns:
            Tuple of (quality_score, is_valid)
            quality_score: 0.0 to 1.0
            is_valid: True if quality > threshold
        """
        if len(channel_data) == 0:
            return 0.0, False

        quality_score = 1.0

        # Check for saturation (assuming microvolts range)
        max_val = np.max(np.abs(channel_data))
        if max_val > 1000:  # Adjust threshold based on Ganglion specs
            quality_score *= 0.5

        # Check for flat signal (disconnection)
        variance = np.var(channel_data)
        if variance < 0.1:  # Too little variation
            quality_score *= 0.3

        # Check for excessive noise
        if variance > 1000:  # Too much variation
            quality_score *= 0.6

        is_valid = quality_score >= self.config.quality_threshold

        return quality_score, is_valid

    def compute_and_create_metrics(
        self,
        data: np.ndarray,
        eeg_channels: List[int],
        session_id: UUID,
        timestamp: datetime,
        computation_method: str = 'welch_psd'
    ) -> Tuple[List, Dict[str, float]]:
        """
        Compute band powers and create MetricsProcessed objects
        Used by both real-time (calibration) and post-session processing

        Args:
            data: Board data array (channels x samples)
            eeg_channels: List of EEG channel indices
            session_id: Session UUID
            timestamp: Timestamp for metrics
            computation_method: Method identifier for database

        Returns:
            Tuple of (list of MetricsProcessed objects, band_powers dict)
        """
        # Import here to avoid circular imports
        try:
            from hero_core.database.models.sensors import MetricsProcessed
        except ImportError:
            logger.error("Could not import MetricsProcessed model")
            return [], {}

        # Compute band powers across all channels
        band_powers = self.compute_multi_channel_band_powers(data, eeg_channels)

        # Create metric objects for each band
        metrics = []
        for band_name, power_value in band_powers.items():
            metric = MetricsProcessed(
                time=timestamp,
                session_id=session_id,
                metric_type=f'eeg_{band_name}',
                value=power_value,
                confidence=None,  # Could add quality score here
                computation_method=computation_method,
                is_valid=True
            )
            metrics.append(metric)

        return metrics, band_powers

    def process_session_data(
        self,
        session_id: UUID,
        db_session: Session,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        Post-session processing: Query raw data and compute metrics

        Args:
            session_id: Session UUID
            db_session: Database session
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Number of processed metric records created
        """
        try:
            from hero_core.database.models.sensors import SensorEEG
        except ImportError:
            logger.error("Could not import database models")
            return 0

        logger.info(f"Starting post-session EEG processing for session {session_id}")

        # Build query
        query = db_session.query(SensorEEG).filter(
            SensorEEG.session_id == session_id
        ).order_by(SensorEEG.time)

        if start_time:
            query = query.filter(SensorEEG.time >= start_time)
        if end_time:
            query = query.filter(SensorEEG.time <= end_time)

        # Fetch all raw samples
        raw_samples = query.all()
        total_samples = len(raw_samples)

        if total_samples == 0:
            logger.warning("No raw EEG data found for session")
            return 0

        logger.info(f"Processing {total_samples} raw EEG samples")

        # Process in chunks
        chunk_size = self.config.processing_interval
        metrics_created = 0

        for i in range(0, total_samples, chunk_size):
            chunk = raw_samples[i:i + chunk_size]

            if len(chunk) < self.psd_size:
                logger.debug(f"Skipping small chunk: {len(chunk)} samples")
                continue

            # Convert to numpy array (channels x samples)
            data = np.array([
                [s.channel_1 for s in chunk],
                [s.channel_2 for s in chunk],
                [s.channel_3 for s in chunk],
                [s.channel_4 for s in chunk],
            ])

            # Use timestamp of middle sample
            timestamp = chunk[len(chunk) // 2].time

            # Use the unified processing method
            metrics, band_powers = self.compute_and_create_metrics(
                data=data,
                eeg_channels=[0, 1, 2, 3],  # Channel indices in our array
                session_id=session_id,
                timestamp=timestamp,
                computation_method='welch_psd_postsession'
            )

            # Add metrics to database
            for metric in metrics:
                db_session.add(metric)
                metrics_created += 1

            # Commit periodically
            if metrics_created % 50 == 0:
                db_session.commit()
                logger.debug(f"Processed {i + len(chunk)}/{total_samples} samples")

        # Final commit
        db_session.commit()
        logger.info(f"✓ Post-session processing complete: {metrics_created} metrics created")

        return metrics_created
