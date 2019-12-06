from qcodes import config, start_all_logging


def _conditionally_start_all_logging():
    def start_logging_on_import() -> bool:
        return (
            config.GUID_components.location != 0 and
            config.GUID_components.work_station != 0 and
            config.telemetry.instrumentation_key != \
                "00000000-0000-0000-0000-000000000000"
            or config.logger.start_logging_on_import
        )

    def running_in_test_or_tool() -> bool:
        import sys
        tools = (
            'pytest.py', 'pytest', '_jb_pytest_runner.py', 'testlauncher.py' )
        return any(sys.argv[0].endswith(tool) for tool in tools)

    if start_logging_on_import() and not running_in_test_or_tool():
        start_all_logging()
