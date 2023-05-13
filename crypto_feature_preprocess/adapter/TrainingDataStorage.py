## Training Data storage
import pandas as pd
import os
from contextlib import ContextDecorator
from ..logging import get_logger

logger = get_logger(__name__)


class TrainingDataStorage(ContextDecorator):
    def __init__(
        self, buffer_size: int, datafile_prefix: str, output_folder: str
    ) -> None:
        self.buffer_size = buffer_size
        self.datafile_prefix = datafile_prefix
        self.output_folder = output_folder
        self.buffer: pd.DataFrame = pd.DataFrame()
        self.buffer_save_counter: int = 0
        # Create output folder if not exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self._written_rows: int = 0
        pass

    def save_data(self, data: pd.DataFrame) -> None:
        # Append data to buffer
        self.buffer = pd.concat([self.buffer, data])

        # Check if buffer is full
        if len(self.buffer) >= self.buffer_size:
            # Save buffer to file
            self.save_buffer()
            # Clear buffer
            self.buffer = pd.DataFrame()
            pass
        pass

    def save_buffer(self) -> None:
        # Save buffer to file
        filename = f"{self.datafile_prefix}_{self.buffer_save_counter}.parquet"
        filepath = os.path.join(self.output_folder, filename)
        self.buffer.to_parquet(filepath)
        self._written_rows += len(self.buffer)
        self.buffer_save_counter += 1
        # Clear buffer
        self.buffer = pd.DataFrame()
        pass

    def save_remaining_data(self) -> None:
        # Save remaining buffer to file
        if len(self.buffer) > 0:
            self.save_buffer()
            pass
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()
        logger.info(
            f"Training data storage closed in {self.output_folder}, total written: {self.written_rows}"
        )
        return False

    @property
    def written_rows(self) -> int:
        return self._written_rows

    def flush(self) -> None:
        self.save_remaining_data()
        pass
