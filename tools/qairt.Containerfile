# QAIRT 2.43 host tools (converter, context-binary generator): Ubuntu 22.04 + python 3.10 + LLVM libc++/libunwind. The SDK itself is mounted at run time.
FROM docker.io/library/ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip libc++1-14 libc++abi1-14 libunwind-14 libatomic1 libgomp1 libpython3.10 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy==1.26.4" "onnx==1.16.1" "protobuf==3.20.3" "pyyaml>=5.3" packaging "typing-extensions>=4.14" "aenum==3.1.15" "mako>=1.1" lxml six "safetensors>=0.4.3" "pydantic==2.8.2" "scipy==1.10.1" psutil rich absl-py attrs jsonschema tabulate decorator invoke joblib pandas
ENV QAIRT_SDK_ROOT=/qairt QNN_SDK_ROOT=/qairt PYTHONPATH=/qairt/lib/python LD_LIBRARY_PATH=/qairt/lib/x86_64-linux-clang:/usr/lib/llvm-14/lib PATH=/qairt/bin/x86_64-linux-clang:/usr/local/bin:/usr/bin:/bin
