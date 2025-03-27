import type {
  operation,
  dtype,
  matinfo,
  MatrixBytes,
  ShardingConfig,
  TPUNodeMetrics,
  TPUMultiNodeMetrics,
  GPUNodeMetrics,
  GPUMultiNodeMetrics,
  RooflineData,
  Metrics
} from './types'

export class Matrix {

  public id: string
  public lhs: matinfo
  public rhs: matinfo
  public type: operation
  public output: { dtype: dtype }

  constructor(type: operation, lhs: matinfo, rhs: matinfo, output: { dtype: dtype }) {
    this.type = type;
    this.lhs = lhs;
    this.rhs = rhs;
    this.output = output;
    this.id = Math.random().toString(36).substr(2, 9);
  }

  calculateFLOPs(): number {
    if (this.type !== "matmul") {
      throw new Error("Only matrix multiplication is supported currently");
    }

    const B = this.lhs.shape[0];
    const D = this.lhs.shape[1];
    const F = this.rhs.shape[1];

    return 2 * B * D * F;
  }

  calculateBytes(): MatrixBytes {
    const bytesPerElement: Record<dtype, number> = {
      bf16: 2,
      fp32: 4,
      int8: 1,
      int4: 0.5,
      fp8: 1
    };

    const lhsBytes = this.lhs.shape.reduce((acc, dim) => acc * dim, 1) *
      bytesPerElement[this.lhs.dtype];

    const rhsBytes = this.rhs.shape.reduce((acc, dim) => acc * dim, 1) *
      bytesPerElement[this.rhs.dtype];

    const B = this.lhs.shape[0];
    const F = this.rhs.shape[1];
    const outputBytes = B * F * bytesPerElement[this.output.dtype];

    return {
      LHSBytes: lhsBytes,
      RHSBytes: rhsBytes,
      outputBytes: outputBytes,
      totalBytes: lhsBytes + rhsBytes + outputBytes
    };
  }

  // Calculate arithmetic intensity
  calculateArithmeticIntensity(): number {
    const bytes = this.calculateBytes();
    const flops = this.calculateFLOPs();

    return flops / bytes.totalBytes;
  }
}

export class TPUv5e {

  // Hardware
  private name: string;
  private type: string;

  // Compute capabilities
  private flopsPerSecondBF16: number;
  private flopsPerSecondINT8: number;

  // Memory specifications
  private hbmCapacity: number;
  private hbmBandwidth: number;
  private vmemCapacity: number;
  private vmemBandwidth: number;

  // Interconnect specifications
  private iciOnewayBandwidth: number;
  private iciBidiBandwidth: number;
  private pcieBandwidth: number;
  private dcnBandwidth: number;

  // Architecture details
  private mxuDimensions: [number, number];
  private coresPerChip: number;
  private iciTopology: string;
  private maxPodSize: [number, number];
  private iciHopLatency: number;

  private matrices: Matrix[];
  private models: any[];

  constructor() {

    this.name = "TPU v5e";
    this.type = "TPU";

    this.flopsPerSecondBF16 = 1.97e14;
    this.flopsPerSecondINT8 = 3.94e14;

    this.hbmCapacity = 16e9;
    this.hbmBandwidth = 8.1e11;
    this.vmemCapacity = 128e6;
    this.vmemBandwidth = 1.78e13;

    this.iciOnewayBandwidth = 4.5e10;
    this.iciBidiBandwidth = 9e10;
    this.pcieBandwidth = 1.5e10;
    this.dcnBandwidth = 2.5e10;

    this.mxuDimensions = [128, 128];
    this.coresPerChip = 1;
    this.iciTopology = "2D";
    this.maxPodSize = [16, 16];
    this.iciHopLatency = 1e-6;

    this.matrices = [];
    this.models = [];
  }

  addMatrix(type: operation, lhs: matinfo, rhs: matinfo, output: { dtype: dtype }): string {
    const matrix = new Matrix(type, lhs, rhs, output);
    this.matrices.push(matrix);
    return matrix.id;
  }

  performanceMetrics(Id: string, mN: boolean = false, sC: ShardingConfig | null = null): Metrics {
    const matrix = this.matrices.find(m => m.id === Id);
    if (!matrix) {
      throw new Error(`Matrix with ID ${Id} not found`);
    }

    if (mN && !sC) {
      throw new Error("Sharding configuration required for multi-node analysis");
    }

    const singleNodeMetrics = this._calculateSingleNodeMetrics(matrix);

    let multiNodeMetrics = null;
    if (mN && sC) {
      multiNodeMetrics = this._calculateMultiNodeMetrics(matrix, sC);
    }

    return {
      singleNode: singleNodeMetrics,
      multiNode: multiNodeMetrics
    };
  }

  _calculateSingleNodeMetrics(matrix: Matrix): TPUNodeMetrics {

    const B = matrix.lhs.shape[0];
    const D = matrix.lhs.shape[1];
    const F = matrix.rhs.shape[1];

    const bytes = matrix.calculateBytes();
    const totalFLOPs = matrix.calculateFLOPs();

    const arithmeticIntensity = matrix.calculateArithmeticIntensity();

    const flopsPerSecond = matrix.lhs.dtype === "int8" ?
      this.flopsPerSecondINT8 :
      this.flopsPerSecondBF16;

    const peakHardwareIntensity = flopsPerSecond / this.hbmBandwidth;

    const isComputeBound = arithmeticIntensity > peakHardwareIntensity;

    const computeTime = totalFLOPs / flopsPerSecond;
    const memoryTime = bytes.totalBytes / this.hbmBandwidth;

    const mxuUtilization = {
      lhsUtilization: (B % this.mxuDimensions[0] === 0) ?
        1.0 :
        (B / Math.ceil(B / this.mxuDimensions[0]) / this.mxuDimensions[0]),

      rhsUtilizationRows: (D % this.mxuDimensions[0] === 0) ?
        1.0 :
        (D / Math.ceil(D / this.mxuDimensions[0]) / this.mxuDimensions[0]),

      rhsUtilizationCols: (F % this.mxuDimensions[1] === 0) ?
        1.0 :
        (F / Math.ceil(F / this.mxuDimensions[1]) / this.mxuDimensions[1])
    };

    const fitsInVMEM = (bytes.LHSBytes + bytes.RHSBytes) <= this.vmemCapacity;
    const vmemComputeTime = computeTime;
    const vmemMemoryTime = bytes.totalBytes / this.vmemBandwidth;
    const vmemTotalTime = Math.max(vmemComputeTime, vmemMemoryTime);

    const vmemMetrics = {
      fitsInVMEM,
      vmemComputeTime,
      vmemMemoryTime,
      vmemTotalTime,
      vmemSpeedup: memoryTime / vmemMemoryTime
    };

    const lowerBoundTime = Math.max(computeTime, memoryTime);
    const upperBoundTime = computeTime + memoryTime;

    return {

      totalFLOPs,
      bytes,
      arithmeticIntensity,
      peakHardwareIntensity,
      isComputeBound,

      computeTime,
      memoryTime,
      lowerBoundTime,
      upperBoundTime,

      mxuUtilization,
      vmemMetrics
    };
  }

  _calculateMultiNodeMetrics(matrix: Matrix, shardingConfig: ShardingConfig): TPUMultiNodeMetrics {

    const shardedMatrix = new Matrix(
      matrix.type,
      JSON.parse(JSON.stringify(matrix.lhs)),
      JSON.parse(JSON.stringify(matrix.rhs)),
      JSON.parse(JSON.stringify(matrix.output))
    );

    if (shardingConfig.lhsShardDim !== null) {
      shardedMatrix.lhs.shape[shardingConfig.lhsShardDim] =
        Math.ceil(matrix.lhs.shape[shardingConfig.lhsShardDim] / shardingConfig.numDevices);
    }

    if (shardingConfig.rhsShardDim !== null) {
      shardedMatrix.rhs.shape[shardingConfig.rhsShardDim] =
        Math.ceil(matrix.rhs.shape[shardingConfig.rhsShardDim] / shardingConfig.numDevices);
    }

    const perDeviceMetrics = this._calculateSingleNodeMetrics(shardedMatrix);
    let communicationCost = 0;

    if (shardingConfig.lhsShardDim === 1 || shardingConfig.rhsShardDim === 0) {
      const B = matrix.lhs.shape[0];
      const F = matrix.rhs.shape[1];
      const outputBytes = B * F *
        (matrix.output.dtype === "bf16" ? 2 :
          matrix.output.dtype === "fp32" ? 4 : 1);

      const steps = Math.log2(shardingConfig.numDevices);
      communicationCost = (outputBytes / 2) * steps / this.iciBidiBandwidth;

    }

    return {
      perDeviceMetrics,
      communicationCost,
      totalTime: Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
      speedupOverSingleDevice: this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
        Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
      shardingEfficiency: perDeviceMetrics.lowerBoundTime /
        Math.max(perDeviceMetrics.lowerBoundTime, communicationCost)
    };
  }

  analyze(matrixId: string): { m: TPUNodeMetrics, r: string[], d: RooflineData } {
    const matrix = this.matrices.find(m => m.id === matrixId);
    if (!matrix) {
      throw new Error(`Matrix with ID ${matrixId} not found`);
    }

    const m = this._calculateSingleNodeMetrics(matrix);
    const r = this._generateRecommendations(matrix, m);
    const d = this._generateRooflineData(matrix, m);

    return { m, r, d };
  }

  _generateRecommendations(matrix: Matrix, metrics: TPUNodeMetrics): string[] {
    const recommendations = [];

    if (!metrics.isComputeBound) {
      recommendations.push("Operation is memory-bound. Consider increasing batch size to improve arithmetic intensity.");

      if (matrix.lhs.dtype === "bf16" && matrix.rhs.dtype === "bf16") {
        recommendations.push("Consider using int8 quantization for weights to reduce memory bandwidth requirements.");
      }

      if (metrics.vmemMetrics.fitsInVMEM) {
        recommendations.push("Consider using VMEM to store weights for higher bandwidth access.");
      }
    }

    if (metrics.mxuUtilization.lhsUtilization < 0.9 ||
      metrics.mxuUtilization.rhsUtilizationRows < 0.9 ||
      metrics.mxuUtilization.rhsUtilizationCols < 0.9) {
      recommendations.push(`Consider padding matrix dimensions to multiples of MXU dimensions (${this.mxuDimensions[0]}x${this.mxuDimensions[1]}) to improve hardware utilization.`);
    }

    const B = matrix.lhs.shape[0];
    if (B < 240 && matrix.lhs.dtype === "bf16") {
      recommendations.push("For bf16 matmul on TPU, batch size should be at least 240 to be compute-bound.");
    }

    if (recommendations.length == 0) { recommendations.push('No recommendations!') }
    return recommendations;
  }

  _generateRooflineData(matrix: Matrix, metrics: TPUNodeMetrics): RooflineData {

    const intensityPoints = [];
    const peakFlops = matrix.lhs.dtype === "int8" ?
      this.flopsPerSecondINT8 :
      this.flopsPerSecondBF16;

    for (let i = -1; i <= 4; i += 0.1) {
      const intensity = Math.pow(10, i);
      const achievableFlops = Math.min(peakFlops, this.hbmBandwidth * intensity);

      intensityPoints.push({
        intensity,
        achievableFlops,
        peakFlops,
        memoryBound: intensity < metrics.peakHardwareIntensity
      });
    }

    const matrixPoint = {
      intensity: metrics.arithmeticIntensity,
      achievableFlops: metrics.isComputeBound ?
        peakFlops :
        this.hbmBandwidth * metrics.arithmeticIntensity,
      isCurrentMatrix: true
    };

    return {
      intensityPoints,
      matrixPoint,
      peakHardwareIntensity: metrics.peakHardwareIntensity,
      peakFlops
    };

  }

}

export class H100 {

  // Hardware
  private name: string;
  private type: string;

  // Compute capabilities
  private flopsPerSecondBF16: number;
  private flopsPerSecondINT8: number;
  private flopsPerSecondBF16Sparse: number;
  private flopsPerSecondFP32: number;
  private flopsPerSecondFP8: number;

  // Memory specifications
  private hbmCapacity: number;
  private hbmBandwidth: number;
  private l2CacheSize: number;
  private sharedMemoryPerSM: number;

  // Interconnect specifications
  private nvlinkBandwidth: number;
  private pcieBandwidth: number;

  // Architecture details
  private tensorCoreConfig: [number, number, number];
  private smCount: number;
  private maxGpusPerNode: number;
  private nvlinkTopology: string;
  private nvlinkLatency: number;
  private pcieLatency: number;

  private matrices: Matrix[];
  private models: any[];

  constructor() {

    this.name = "NVIDIA H100 SXM5";
    this.type = "GPU";

    this.flopsPerSecondBF16 = 9.89e14;
    this.flopsPerSecondBF16Sparse = 1.979e15;
    this.flopsPerSecondFP32 = 6.7e13;
    this.flopsPerSecondINT8 = 1.979e15;
    this.flopsPerSecondFP8 = 3.958e15;

    this.hbmCapacity = 80e9;
    this.hbmBandwidth = 3.35e12;
    this.l2CacheSize = 50e6;
    this.sharedMemoryPerSM = 228e3;

    this.nvlinkBandwidth = 9e10 * 18;
    this.pcieBandwidth = 8e10;

    this.tensorCoreConfig = [4, 4, 16];
    this.smCount = 132;
    this.maxGpusPerNode = 256;
    this.nvlinkTopology = "all-to-all";
    this.nvlinkLatency = 0.5e-6;
    this.pcieLatency = 2e-6;

    this.matrices = [];
    this.models = [];
  }

  addMatrix(type: operation, lhs: matinfo, rhs: matinfo, output: { dtype: dtype }) {
    const matrix = new Matrix(type, lhs, rhs, output);
    this.matrices.push(matrix);
    return matrix.id;
  }

  performanceMetrics(matrixId: string, multiNode: boolean = false, shardingConfig: ShardingConfig | null = null): Metrics {

    const matrix = this.matrices.find(m => m.id === matrixId);
    if (!matrix) {
      throw new Error(`Matrix with ID ${matrixId} not found`);
    }

    if (multiNode && !shardingConfig) {
      throw new Error("Sharding configuration required for multi-node analysis");
    }

    const singleNodeMetrics = this._calculateSingleNodeMetrics(matrix);

    let multiNodeMetrics = null;
    if (multiNode && shardingConfig) {
      multiNodeMetrics = this._calculateMultiNodeMetrics(matrix, shardingConfig);
    }

    return {
      singleNode: singleNodeMetrics,
      multiNode: multiNodeMetrics
    };
  }

  _calculateSingleNodeMetrics(matrix: Matrix): GPUNodeMetrics {
    const B = matrix.lhs.shape[0];
    const D = matrix.lhs.shape[1];
    const F = matrix.rhs.shape[1];

    const bytes = matrix.calculateBytes();
    const totalFLOPs = matrix.calculateFLOPs();

    const arithmeticIntensity = matrix.calculateArithmeticIntensity();

    let flopsPerSecond;
    switch (matrix.lhs.dtype) {
      case "bf16":
        flopsPerSecond = this.flopsPerSecondBF16;
        break;
      case "fp32":
        flopsPerSecond = this.flopsPerSecondFP32;
        break;
      case "int8":
        flopsPerSecond = this.flopsPerSecondINT8;
        break;
      case "fp8":
        flopsPerSecond = this.flopsPerSecondFP8;
        break;
      default:
        flopsPerSecond = this.flopsPerSecondBF16;
    }

    const peakHardwareIntensity = flopsPerSecond / this.hbmBandwidth;
    const isComputeBound = arithmeticIntensity > peakHardwareIntensity;
    const computeTime = totalFLOPs / flopsPerSecond;
    const memoryTime = bytes.totalBytes / this.hbmBandwidth;

    const tcM = this.tensorCoreConfig[0];
    const tcN = this.tensorCoreConfig[1];
    const tcK = this.tensorCoreConfig[2];

    const tensorCoreUtilization = {
      mUtilization: (B % tcM === 0) ? 1.0 : (B / Math.ceil(B / tcM) / tcM),
      nUtilization: (F % tcN === 0) ? 1.0 : (F / Math.ceil(F / tcN) / tcN),
      kUtilization: (D % tcK === 0) ? 1.0 : (D / Math.ceil(D / tcK) / tcK)
    };

    const warpsPerSM = 64;
    const threadsPerWarp = 32;
    const threadsPerBlock = 256;
    const blocksPerSM = Math.min(16, Math.floor(warpsPerSM * threadsPerWarp / threadsPerBlock));

    const blocksNeeded = Math.ceil(B / 32) * Math.ceil(F / 32);
    const smOccupancy = Math.min(1.0, blocksNeeded / (blocksPerSM * this.smCount));

    const canFitWeightsInL2 = bytes.RHSBytes <= this.l2CacheSize;
    const l2CacheBenefit = canFitWeightsInL2 ? 1.5 : 1.0;

    const effectiveMemoryTime = memoryTime / l2CacheBenefit;
    const lowerBoundTime = Math.max(computeTime, effectiveMemoryTime);
    const upperBoundTime = computeTime + effectiveMemoryTime;

    return {
      totalFLOPs,
      bytes,
      arithmeticIntensity,
      peakHardwareIntensity,
      isComputeBound,

      computeTime,
      memoryTime,
      effectiveMemoryTime,
      lowerBoundTime,
      upperBoundTime,

      tensorCoreUtilization,
      smOccupancy,
      canFitWeightsInL2,
      l2CacheBenefit
    };
  }

  _calculateMultiNodeMetrics(matrix: Matrix, shardingConfig: ShardingConfig): GPUMultiNodeMetrics {

    const shardedMatrix = new Matrix(
      matrix.type,
      JSON.parse(JSON.stringify(matrix.lhs)),
      JSON.parse(JSON.stringify(matrix.rhs)),
      JSON.parse(JSON.stringify(matrix.output))
    );

    if (shardingConfig.lhsShardDim !== null) {
      shardedMatrix.lhs.shape[shardingConfig.lhsShardDim] =
        Math.ceil(matrix.lhs.shape[shardingConfig.lhsShardDim] / shardingConfig.numDevices);
    }

    if (shardingConfig.rhsShardDim !== null) {
      shardedMatrix.rhs.shape[shardingConfig.rhsShardDim] =
        Math.ceil(matrix.rhs.shape[shardingConfig.rhsShardDim] / shardingConfig.numDevices);
    }

    const perDeviceMetrics = this._calculateSingleNodeMetrics(shardedMatrix);

    let communicationCost = 0;

    if (shardingConfig.lhsShardDim === 1 || shardingConfig.rhsShardDim === 0) {
      const B = matrix.lhs.shape[0];
      const F = matrix.rhs.shape[1];
      const outputBytes = B * F *
        (matrix.output.dtype === "bf16" ? 2 :
          matrix.output.dtype === "fp32" ? 4 : 1);

      // For H100 with NVSwitch, communication is more efficient
      if (shardingConfig.numDevices <= this.maxGpusPerNode) {
        // Within a single NVSwitch domain - more efficient communication
        const steps = Math.log2(shardingConfig.numDevices);
        communicationCost = (outputBytes / 2) * steps / this.nvlinkBandwidth;
      } else {
        // Across multiple nodes - less efficient
        const nodesNeeded = Math.ceil(shardingConfig.numDevices / this.maxGpusPerNode);
        const intraNodeSteps = Math.log2(this.maxGpusPerNode);
        const interNodeSteps = Math.log2(nodesNeeded);

        const intraNodeCost = (outputBytes / 2) * intraNodeSteps / this.nvlinkBandwidth;
        const interNodeCost = (outputBytes / nodesNeeded) * interNodeSteps / this.pcieBandwidth;

        communicationCost = intraNodeCost + interNodeCost;
      }
    }

    return {
      perDeviceMetrics,
      communicationCost,
      totalTime: Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
      speedupOverSingleDevice: this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
        Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
      shardingEfficiency: perDeviceMetrics.lowerBoundTime /
        Math.max(perDeviceMetrics.lowerBoundTime, communicationCost),
      scalingEfficiency: (this._calculateSingleNodeMetrics(matrix).lowerBoundTime /
        Math.max(perDeviceMetrics.lowerBoundTime, communicationCost)) /
        shardingConfig.numDevices
    };
  }

  analyze(matrixId: string): { m: GPUNodeMetrics, r: string[], d: RooflineData } {
    const matrix = this.matrices.find(m => m.id === matrixId);
    if (!matrix) {
      throw new Error(`Matrix with ID ${matrixId} not found`);
    }

    const m = this._calculateSingleNodeMetrics(matrix);
    const r = this._generateRecommendations(matrix, m);
    const d = this._generateRooflineData(matrix, m);

    return {
      m,
      r,
      d
    };
  }

  _generateRecommendations(matrix: Matrix, metrics: GPUNodeMetrics): string[] {
    const recommendations = [];

    if (!metrics.isComputeBound) {
      recommendations.push("Operation is memory-bound. Consider increasing batch size to improve arithmetic intensity.");

      if (matrix.lhs.dtype === "bf16" && matrix.rhs.dtype === "bf16") {
        recommendations.push("Consider using FP8 precision to reduce memory bandwidth requirements and increase compute throughput.");
      }

      recommendations.push("Consider using structured sparsity to potentially double compute throughput.");
    }

    if (metrics.tensorCoreUtilization.mUtilization < 0.9 ||
      metrics.tensorCoreUtilization.nUtilization < 0.9 ||
      metrics.tensorCoreUtilization.kUtilization < 0.9) {
      recommendations.push(`Consider padding matrix dimensions to multiples of tensor core dimensions (${this.tensorCoreConfig.join('x')}) to improve hardware utilization.`);
    }

    const B = matrix.lhs.shape[0];
    if (B < 300 && matrix.lhs.dtype === "bf16") {
      recommendations.push("For bf16 matmul on H100, batch size should be at least 300 to be compute-bound.");
    }

    if (B % 32 !== 0 || matrix.rhs.shape[1] % 32 !== 0 || matrix.lhs.shape[1] % 16 !== 0) {
      recommendations.push("For optimal CUDA kernel performance, consider using dimensions that are multiples of 32 for M and N, and 16 for K.");
    }

    if (recommendations.length == 0) { recommendations.push('No recommendations!') }
    return recommendations;
  }

  _generateRooflineData(matrix: Matrix, metrics: GPUNodeMetrics): RooflineData {

    let flopsPerSecond;
    switch (matrix.lhs.dtype) {
      case "bf16":
        flopsPerSecond = this.flopsPerSecondBF16;
        break;
      case "fp32":
        flopsPerSecond = this.flopsPerSecondFP32;
        break;
      case "int8":
        flopsPerSecond = this.flopsPerSecondINT8;
        break;
      case "fp8":
        flopsPerSecond = this.flopsPerSecondFP8;
        break;
      default:
        flopsPerSecond = this.flopsPerSecondBF16;
    }

    const intensityPoints = [];

    for (let i = -1; i <= 4; i += 0.1) {
      const intensity = Math.pow(10, i);
      const achievableFlops = Math.min(flopsPerSecond, this.hbmBandwidth * intensity);

      intensityPoints.push({
        intensity,
        achievableFlops,
        peakFlops: flopsPerSecond,
        memoryBound: intensity < metrics.peakHardwareIntensity
      });
    }

    const matrixPoint = {
      intensity: metrics.arithmeticIntensity,
      achievableFlops: metrics.isComputeBound ?
        flopsPerSecond :
        this.hbmBandwidth * metrics.arithmeticIntensity,
      isCurrentMatrix: true
    };

    return {
      intensityPoints,
      matrixPoint,
      peakHardwareIntensity: metrics.peakHardwareIntensity,
      peakFlops: flopsPerSecond
    };
  }
  
}