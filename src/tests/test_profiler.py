import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
# sys.path.append('./hpc-profiler/hpc-profiler')  # Use the path to the directory of your Python file

import Profiler

class TestProfiler(unittest.TestCase):
    @patch('Profiler.subprocess.Popen')
    @patch('Profiler.psutil')
    @patch('Profiler.threading.Thread')
    @patch('builtins.open', new_callable=mock_open)
    def test_monitor_system_utilization(self, mock_file, mock_thread, mock_psutil, mock_popen):
        # Setup mock for psutil
        mock_psutil.cpu_percent.return_value = 10
        mock_psutil.virtual_memory.return_value = type('obj', (object,), {'percent': 50, 'used': 1024**3 * 1.5})  # 1.5 GB

        # Setup mock for subprocess.Popen
        mock_popen.return_value = MagicMock()

        # Call the function
        Profiler.monitor_system_utilization(benchmark_time_interval=1)

        # Check if subprocess.Popen was called correctly
        mock_popen.assert_called_once_with('gpustat -a -i 30 > gpustat_GPT2_Large.log 2>&1 &', shell=True)

        # Check if file was opened correctly for writing initial headers
        mock_file.assert_called_with('GPT2_Large/Graphs/GPT2_Large_CPU_RAM_Utilization.csv', mode='w', newline='')

        # Check if thread was started
        mock_thread.assert_called_once()

        # Check if the thread target function is correct
        args, kwargs = mock_thread.call_args
        self.assertEqual(kwargs['target'].__name__, 'monitor_system_utilization_helper')
        self.assertEqual(kwargs['args'][0], 1)  # Ensure the interval passed is correct

if __name__ == '__main__':
    unittest.main()