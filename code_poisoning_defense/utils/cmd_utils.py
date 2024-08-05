from utils.logger import logger
import time
import subprocess

def run_cmd_with_output(cmd):
    start_time = time.time()
    logger.info(f"cmd to run: {cmd}")
    p = subprocess.run(cmd,
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    try:
        output = p.stdout.decode('utf-8')
    except UnicodeDecodeError:
        logger.warn("cmd UnicodeDecodeError")
        output = p.stdout.decode('unicode_escape')
    
    error = p.stderr.decode('utf-8')
    # if len(error) > 0 and error.strip(
    # ) != "Picked up JAVA_TOOL_OPTIONS: -Dfile.encoding=UTF8":
    #     logger.error(f"output error: {error}")

    if len(output) > 0:
        logger.debug(f"output of this cmd: {output}")

    logger.debug(f"cmd execution time: {time.time() - start_time}")
    return output, error