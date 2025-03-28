export const enum dtype {
    fp8 = 'fp8',
    bf16 = 'bf16',
    fp32 = 'fp32',
    int8 = 'int8',
    int4 = 'int4'
}

export const enum operation {
    matmul = 'matmul'
}

export interface matinfo {
    shape: [number, number]
    dtype: dtype
}

export interface Hardware {
    Name: string
    Type: "TPU" | "GPU"
}

export interface MatrixBytes {
    LHSBytes: number,
    RHSBytes: number,
    outputBytes: number,
    totalBytes: number
}

export interface ShardingConfig {
    numDevices: number
    lhsShardDim: number
    rhsShardDim: number
}

export interface TPUNodeMetrics {

    totalFLOPs: number
    bytes: MatrixBytes
    arithmeticIntensity: number
    peakHardwareIntensity: number
    isComputeBound: boolean

    computeTime: number
    memoryTime: number
    lowerBoundTime: number
    upperBoundTime: number

    mxuUtilization: {
        lhsUtilization: number;
        rhsUtilizationRows: number;
        rhsUtilizationCols: number;
    }

    vmemMetrics: {
        fitsInVMEM: boolean;
        vmemComputeTime: number;
        vmemMemoryTime: number;
        vmemTotalTime: number;
        vmemSpeedup: number;
    }

}

export interface TPUMultiNodeMetrics {
    perDeviceMetrics: TPUNodeMetrics
    communicationCost: number
    totalTime: number
    speedupOverSingleDevice: number
    shardingEfficiency: number
}

export interface GPUNodeMetrics {

    totalFLOPs: number,
    bytes: MatrixBytes,
    arithmeticIntensity: number,
    peakHardwareIntensity: number,
    isComputeBound: boolean,

    computeTime: number
    memoryTime: number
    effectiveMemoryTime: number
    lowerBoundTime: number
    upperBoundTime: number

    tensorCoreUtilization: {
        mUtilization: number;
        nUtilization: number;
        kUtilization: number;
    }

    smOccupancy: number
    canFitWeightsInL2: boolean
    l2CacheBenefit: number
    
}

export interface GPUMultiNodeMetrics {
    perDeviceMetrics: GPUNodeMetrics
    communicationCost: number
    totalTime: number
    speedupOverSingleDevice: number
    shardingEfficiency: number
    scalingEfficiency: number
}

export type singleNodeMetrics = GPUNodeMetrics | TPUNodeMetrics
export type multiNodeMetrics = GPUMultiNodeMetrics | TPUMultiNodeMetrics


export interface RooflineData {
    intensityPoints: {
        intensity: number;
        achievableFlops: number;
        peakFlops: number;
        memoryBound: boolean;
    }[];
    matrixPoint: {
        intensity: number;
        achievableFlops: number;
        isCurrentMatrix: boolean;
    };
    peakHardwareIntensity: number;
    peakFlops: number;
}