"""
ZSim 和 Ramulator 执行器模块

用于在 GCN 调度器中将任务分配到 PIM/PNM 时，实际调用 ZSim 生成 trace 文件，
并使用 Ramulator 进行仿真。
"""

import os
import subprocess
import tempfile
import shutil
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZSimRamulatorExecutor:
    """ZSim 和 Ramulator 执行器"""
    
    def __init__(
        self,
        zsim_path: Optional[str] = None,
        ramulator_path: Optional[str] = None,
        zsim_config_template: Optional[str] = None,
        ramulator_config: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_docker: bool = False,
        docker_image: str = "ramulator-pim:latest",
        docker_workspace: str = "/workspace"
    ):
        """
        初始化执行器
        
        Args:
            zsim_path: ZSim 可执行文件路径（如果 use_docker=False）
            ramulator_path: Ramulator 可执行文件路径（如果 use_docker=False）
            zsim_config_template: ZSim 配置文件模板路径
            ramulator_config: Ramulator 配置文件路径
            output_dir: 输出目录（trace 文件和统计文件）
            use_docker: 是否使用 Docker 容器运行
            docker_image: Docker 镜像名称
            docker_workspace: Docker 容器内的工作目录
        """
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_workspace = docker_workspace
        
        if use_docker:
            # Docker 模式：使用容器内的路径
            self.zsim_path = f"{docker_workspace}/zsim-ramulator/build/opt/zsim"
            self.ramulator_path = f"{docker_workspace}/ramulator/ramulator"
            self.zsim_config_template = zsim_config_template or f"{docker_workspace}/zsim-ramulator/tests/host.cfg"
            self.ramulator_config = ramulator_config or f"{docker_workspace}/ramulator/Configs/host.cfg"
        else:
            # 本地模式：使用提供的路径或默认路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ramulator_base = os.path.join(base_dir, "..", "..", "ramulator-pim-master")
            
            self.zsim_path = zsim_path or os.path.join(ramulator_base, "zsim-ramulator", "build", "opt", "zsim")
            self.ramulator_path = ramulator_path or os.path.join(ramulator_base, "ramulator", "ramulator")
            self.zsim_config_template = zsim_config_template or os.path.join(ramulator_base, "zsim-ramulator", "tests", "host.cfg")
            self.ramulator_config = ramulator_config or os.path.join(ramulator_base, "ramulator", "Configs", "host.cfg")
        
        # 输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(base_dir, "results", "zsim_ramulator_traces")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ZSimRamulatorExecutor initialized:")
        logger.info(f"  - Use Docker: {use_docker}")
        logger.info(f"  - ZSim path: {self.zsim_path}")
        logger.info(f"  - Ramulator path: {self.ramulator_path}")
        logger.info(f"  - Output dir: {self.output_dir}")
    
    def _run_docker_command(self, command: str, workdir: str = None) -> Tuple[int, str, str]:
        """
        在 Docker 容器中运行命令
        
        Args:
            command: 要执行的命令
            workdir: 工作目录
            
        Returns:
            (returncode, stdout, stderr)
        """
        docker_cmd = [
            "docker", "run", "--rm",
            "--security-opt", "seccomp=unconfined",
            "-v", f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/ramulator-pim-master:{self.docker_workspace}",
            self.docker_image,
            "bash", "-c", command
        ]
        
        if workdir:
            docker_cmd.insert(-2, "-w")
            docker_cmd.insert(-2, workdir)
        
        logger.debug(f"Running Docker command: {' '.join(docker_cmd)}")
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        return result.returncode, result.stdout, result.stderr
    
    def _run_local_command(self, command: list, cwd: str = None) -> Tuple[int, str, str]:
        """
        在本地运行命令
        
        Args:
            command: 命令列表
            cwd: 工作目录
            
        Returns:
            (returncode, stdout, stderr)
        """
        logger.debug(f"Running local command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        return result.returncode, result.stdout, result.stderr
    
    def generate_zsim_config(
        self,
        task_id: str,
        trace_filename: str,
        python_script: str,
        python_args: list = None,
        cores: int = 1,
        split_trace: bool = False
    ) -> str:
        """
        生成 ZSim 配置文件
        
        Args:
            task_id: 任务 ID（用于唯一标识）
            trace_filename: trace 输出文件名
            python_script: 要运行的 Python 脚本路径
            python_args: Python 脚本的参数列表
            cores: CPU 核心数
            split_trace: 是否生成 per-core trace 文件
            
        Returns:
            生成的配置文件路径
        """
        # 读取模板
        if self.use_docker:
            # Docker 模式：配置文件在容器内
            config_path = os.path.join(self.output_dir, f"zsim_config_{task_id}.cfg")
            # 需要从容器内读取模板，这里简化处理，直接生成
            template_path = self.zsim_config_template
        else:
            template_path = self.zsim_config_template
            config_path = os.path.join(self.output_dir, f"zsim_config_{task_id}.cfg")
        
        # 读取模板内容
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                config_content = f.read()
        else:
            # 如果模板不存在，使用默认配置
            logger.warning(f"Template {template_path} not found, using default config")
            config_content = self._get_default_zsim_config()
        
        # 构建 Python 命令
        python_cmd = f"python3 {python_script}"
        if python_args:
            python_cmd += " " + " ".join(str(arg) for arg in python_args)
        
        # 替换配置中的关键字段
        config_content = config_content.replace(
            'cores = 4;',
            f'cores = {cores};'
        )
        config_content = config_content.replace(
            'outFile = "rodiniaBFS.out"',
            f'outFile = "{trace_filename}"'
        )
        config_content = config_content.replace(
            'splitTrace = true;',
            f'splitTrace = {"true" if split_trace else "false"};'
        )
        
        # 替换 process0.command
        import re
        pattern = r'process0\s*=\s*\{[^}]*command\s*=\s*"[^"]*"'
        replacement = f'process0 = {{\n    command = "{python_cmd}"'
        config_content = re.sub(pattern, replacement, config_content, flags=re.MULTILINE)
        
        # 如果 process0 不存在，添加它
        if 'process0 =' not in config_content:
            config_content += f'\n\nprocess0 = {{\n    command = "{python_cmd}"\n    startFastForwarded = True;\n}};\n'
        
        # 写入配置文件
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Generated ZSim config: {config_path}")
        return config_path
    
    def _get_default_zsim_config(self) -> str:
        """返回默认的 ZSim 配置"""
        return """sys = {
    lineSize = 64;
    frequency = 2400;

    cores = {
        core = {
            type = "OOO";
            cores = 1;
            icache = "l1i";
            dcache = "l1d";
        };
    };

    caches = {
        l1d = {
            caches = 1;
            size = 32768;
            array = {
                type = "SetAssoc";
                ways = 8;
            };
            latency = 4;
        };

        l1i = {
            caches = 1;
            size = 32768;
            array = {
                type = "SetAssoc";
                ways = 4;
            };
            latency = 3;
        };

        l2 = {
            caches = 1;
            size = 262144;
            latency = 7;
            array = {
                type = "SetAssoc";
                ways = 8;
            };
            children = "l1i|l1d";
        };

        l3 = {
            caches = 1;
            banks = 4;
            size = 8388608;
            latency = 27;
            array = {
                type = "SetAssoc";
                hash = "H3";
                ways = 16;
            };
            children = "l2";
        };
    };

    mem = {
        type = "Traces";
        only_offload = true;
        pim_traces = false;
        instr_traces = true;        
        outFile = "gcn_trace.out"
    };
};

sim = {
    phaseLength = 10000;
    max_offload_instrs = 1000000000L;
    statsPhaseInterval = 1000;
    printHierarchy = true;
    splitTrace = false;
};

process0 = {
    command = "python3 /path/to/script.py"
    startFastForwarded = True;
};
"""
    
    def run_zsim(self, config_path: str, trace_output_dir: str = None) -> Tuple[bool, str]:
        """
        运行 ZSim 生成 trace 文件
        
        Args:
            config_path: ZSim 配置文件路径
            trace_output_dir: trace 文件输出目录
            
        Returns:
            (success, trace_file_path)
        """
        if trace_output_dir is None:
            trace_output_dir = self.output_dir
        
        if self.use_docker:
            # Docker 模式
            # 将配置文件路径转换为容器内路径
            container_config = config_path.replace(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/ramulator-pim-master",
                self.docker_workspace
            )
            
            # 获取 trace 文件名（从配置中解析）
            with open(config_path, 'r') as f:
                config_content = f.read()
            import re
            match = re.search(r'outFile\s*=\s*"([^"]+)"', config_content)
            trace_filename = match.group(1) if match else "gcn_trace.out"
            
            container_trace_dir = f"{self.docker_workspace}/zsim-ramulator"
            command = f"cd {self.docker_workspace}/zsim-ramulator && rm -f {trace_filename}* && {self.zsim_path} {container_config}"
            
            returncode, stdout, stderr = self._run_docker_command(command)
            
            if returncode == 0:
                # 从容器复制 trace 文件到本地
                trace_path = os.path.join(trace_output_dir, trace_filename)
                # 注意：Docker 模式下，文件已经在挂载的目录中，可以直接访问
                # 但需要确保路径正确
                actual_trace_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "ramulator-pim-master",
                    "zsim-ramulator",
                    trace_filename
                )
                if os.path.exists(actual_trace_path):
                    shutil.copy2(actual_trace_path, trace_path)
                    logger.info(f"ZSim trace generated: {trace_path}")
                    return True, trace_path
                else:
                    logger.error(f"Trace file not found at {actual_trace_path}")
                    return False, ""
            else:
                logger.error(f"ZSim failed: {stderr}")
                return False, ""
        else:
            # 本地模式
            zsim_dir = os.path.dirname(self.zsim_path)
            command = [self.zsim_path, config_path]
            
            returncode, stdout, stderr = self._run_local_command(command, cwd=zsim_dir)
            
            if returncode == 0:
                # 查找生成的 trace 文件
                # 从配置中获取 trace 文件名
                with open(config_path, 'r') as f:
                    config_content = f.read()
                import re
                match = re.search(r'outFile\s*=\s*"([^"]+)"', config_content)
                trace_filename = match.group(1) if match else "gcn_trace.out"
                
                trace_path = os.path.join(zsim_dir, trace_filename)
                if os.path.exists(trace_path):
                    # 复制到输出目录
                    output_trace_path = os.path.join(trace_output_dir, trace_filename)
                    shutil.copy2(trace_path, output_trace_path)
                    logger.info(f"ZSim trace generated: {output_trace_path}")
                    return True, output_trace_path
                else:
                    logger.error(f"Trace file not found at {trace_path}")
                    return False, ""
            else:
                logger.error(f"ZSim failed: {stderr}")
                return False, ""
    
    def run_ramulator(
        self,
        trace_path: str,
        stats_output: str = None,
        mode: str = "cpu",
        cores: int = 1,
        split_trace: bool = False
    ) -> Tuple[bool, str]:
        """
        运行 Ramulator 进行仿真
        
        Args:
            trace_path: ZSim trace 文件路径
            stats_output: 统计输出文件路径
            mode: 仿真模式（"cpu" 或 "pim"）
            cores: CPU 核心数
            split_trace: 是否使用多核 trace 分片
            
        Returns:
            (success, stats_file_path)
        """
        if stats_output is None:
            stats_output = os.path.join(self.output_dir, f"ramulator_stats_{os.path.basename(trace_path)}.stats")
        
        if self.use_docker:
            # Docker 模式
            container_trace = trace_path.replace(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/ramulator-pim-master",
                self.docker_workspace
            )
            container_stats = f"{self.docker_workspace}/data/{os.path.basename(stats_output)}"
            container_config = self.ramulator_config
            
            split_flag = "--split-trace=true" if split_trace else ""
            command = (
                f"cd {self.docker_workspace}/ramulator && "
                f"{self.ramulator_path} --config {container_config} "
                f"--mode={mode} --stats {container_stats} "
                f"--trace {container_trace} "
                f"--core-org=outOrder --number-cores={cores} "
                f"--trace-format=zsim {split_flag}"
            )
            
            returncode, stdout, stderr = self._run_docker_command(command)
            
            if returncode == 0:
                # 从容器复制统计文件
                actual_stats_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "ramulator-pim-master",
                    "data",
                    os.path.basename(stats_output)
                )
                if os.path.exists(actual_stats_path):
                    shutil.copy2(actual_stats_path, stats_output)
                    logger.info(f"Ramulator stats generated: {stats_output}")
                    return True, stats_output
                else:
                    logger.warning(f"Stats file not found, but Ramulator succeeded")
                    return True, stats_output  # 仍然返回成功，因为 Ramulator 执行成功
            else:
                logger.error(f"Ramulator failed: {stderr}")
                return False, ""
        else:
            # 本地模式
            ramulator_dir = os.path.dirname(self.ramulator_path)
            command = [
                self.ramulator_path,
                "--config", self.ramulator_config,
                "--mode", mode,
                "--stats", stats_output,
                "--trace", trace_path,
                "--core-org", "outOrder",
                "--number-cores", str(cores),
                "--trace-format", "zsim"
            ]
            if split_trace:
                command.append("--split-trace=true")
            
            returncode, stdout, stderr = self._run_local_command(command, cwd=ramulator_dir)
            
            if returncode == 0:
                if os.path.exists(stats_output):
                    logger.info(f"Ramulator stats generated: {stats_output}")
                    return True, stats_output
                else:
                    logger.warning(f"Stats file not found, but Ramulator succeeded")
                    return True, stats_output
            else:
                logger.error(f"Ramulator failed: {stderr}")
                return False, ""
    
    def execute_task(
        self,
        task_id: str,
        device: str,
        phase: str,
        layer: int,
        partition_id: int,
        epoch: int,
        python_script_content: str = None,
        python_script_path: str = None,
        python_args: list = None
    ) -> Dict:
        """
        执行完整的任务流程：生成配置 -> 运行 ZSim -> 运行 Ramulator
        
        Args:
            task_id: 任务唯一标识
            device: 设备类型（"PIM" 或 "PNM"）
            phase: 阶段（"AGG" 或 "UPDATE"）
            layer: 层编号（1 或 2）
            partition_id: 分区 ID
            epoch: epoch 编号
            python_script_content: Python 脚本内容（如果提供，会写入临时文件）
            python_script_path: Python 脚本路径（如果提供）
            python_args: Python 脚本参数
            
        Returns:
            执行结果字典，包含 success, trace_path, stats_path 等信息
        """
        logger.info(f"Executing task {task_id} on {device} (Phase: {phase}, Layer: {layer}, Partition: {partition_id}, Epoch: {epoch})")
        
        # 生成唯一的 trace 文件名
        trace_filename = f"gcn_{device}_{phase}_L{layer}_P{partition_id}_E{epoch}_{task_id}.out"
        
        # 处理 Python 脚本
        if python_script_content:
            # 写入临时文件
            script_path = os.path.join(self.output_dir, f"gcn_task_{task_id}.py")
            with open(script_path, 'w') as f:
                f.write(python_script_content)
            python_script = script_path
        elif python_script_path:
            python_script = python_script_path
        else:
            logger.error("Either python_script_content or python_script_path must be provided")
            return {"success": False, "error": "No Python script provided"}
        
        # 生成 ZSim 配置
        config_path = self.generate_zsim_config(
            task_id=task_id,
            trace_filename=trace_filename,
            python_script=python_script,
            python_args=python_args,
            cores=1,
            split_trace=False
        )
        
        # 运行 ZSim
        zsim_success, trace_path = self.run_zsim(config_path)
        
        if not zsim_success:
            return {
                "success": False,
                "error": "ZSim execution failed",
                "task_id": task_id,
                "device": device,
                "phase": phase
            }
        
        # 运行 Ramulator
        ramulator_success, stats_path = self.run_ramulator(
            trace_path=trace_path,
            mode="cpu" if device == "PNM" else "pim",  # PIM 模式用于 PIM 设备
            cores=1,
            split_trace=False
        )
        
        return {
            "success": ramulator_success,
            "task_id": task_id,
            "device": device,
            "phase": phase,
            "layer": layer,
            "partition_id": partition_id,
            "epoch": epoch,
            "trace_path": trace_path,
            "stats_path": stats_path,
            "config_path": config_path
        }

