# パッケージゾーン
println("calculating IQ")
using Falcons
using ReferenceFrameRotations
using Healpix
using Plots
using NPZ
using LinearAlgebra
using StatsBase
using BenchmarkTools
using PyCall
using TickTock
using Base.Threads
using StaticArrays
using PyPlot
using Formatting
using Statistics
using ProgressMeter
hp = pyimport("healpy");
np = pyimport("numpy"); 
println("パッケージゾーン終わり")
# パッケージゾーン終わり

# インプットゾーン
type_array = [Int,Int,Int,Float64,Float64,Int]　# ARGSはstringなのでこれで変換するかたしてい
input_array = ARGS
println(input_array)
const input1, input2, input3, input4, input5, input6 = parse.(type_array, input_array)# 型変換してる

const day = 60 * 60 * 24
const year = day * 365;
duration_period = input6 * year
const NSIDE = input1
const samp_rate = input2
const prec_period = 192.348
const division = input3
const delta_theta = input4 
const error_drection = input5
lmax = 3 * NSIDE - 1
resol = Resolution(NSIDE)
npix = nside2npix(NSIDE)

state_array = "pixwin_smooth0.5deg_" * string(NSIDE)
output_state = "deltaTHETA_" * string(delta_theta) * "_errDirect_" * string(error_drection) * "_" * state_array
data_dir = "/group/cmb/litebird/usr/naoyadoi/data/" ;
println("インプットゾーン終わり")
# インプットゾーン終わり

# 関数定義ゾーン
function calc_pix_tod(theta_tod, phi_tod, resol::Resolution) 
    @views pix_tod = zeros(Int, size(theta_tod)[1])
    @views @inbounds @threads for i in 1:size(phi_tod)[1]
        @views pix_tod[i] = ang2pixRing(resol, theta_tod[i], phi_tod[i])
    end
    return pix_tod
end

function calc_d_tod(pix_tod, psi_tod, hwp_tod) 
    @views d_tod = zeros(size(pix_tod)[1], 3)

    @views A =  cos.(2 * psi_tod)
    @views B = sin.(2 * psi_tod)
    @views C = cos.(4 * hwp_tod)
    @views D = sin.(4 * hwp_tod)
    @views d_tod[:, 1] .= 1
    @views d_tod[:, 2] = @. A * C + D * B
    @views d_tod[:, 3] = @. A * D - B * C 
    return d_tod ./ 2
end

function calc_p_tod(psi_tod, hwp_tod, IQU, theta_tod, phi_tod)
    @views pix_tod = calc_pix_tod(theta_tod, phi_tod, resol::Resolution); # true data
    @views p_tod = zeros(size(pix_tod)[1])
    @views IQU_tod = zeros(size(pix_tod)[1], 3)
    @views d_tod = calc_d_tod(pix_tod, psi_tod, hwp_tod) 

    for i in 1:3
        IQU_tod[:,i] = hp.get_interp_val(IQU[:,i], theta_tod, phi_tod, lonlat=false)
    end
    @views @inbounds @threads for i in 1:size(pix_tod)[1]
        p_tod[i] = dot(IQU_tod[i,:], d_tod[i,:])
    end
    return p_tod 
end

function calc_IQU_from_tod(ss_true::ScanningStrategy,　division::Int, D_matrix, IQU)　　# DのピクセルにハーフウェブプレートのTOD、ψのTOD
    section = Int(ss_true.duration / division)
    dp_map = zeros((npix, 3)) 
    IQU_out = zeros((npix, 3)) 
    @views @inbounds for j = 1:division
        BEGIN = (j - 1) * section
        END = j * section 
        
        @views theta_tod, phi_tod, psi_tod, time_array = get_pointings_tuple(ss_true, BEGIN, END);
        @views pix_tod = calc_pix_tod(theta_tod, phi_tod, resol::Resolution); # true data
        @views hwp_tod = @. 2π * ((ss_true.hwp_rpm * time_array / 60) % (1))
        @views d_tod_true = calc_d_tod(pix_tod, psi_tod, hwp_tod) # true d_tod
        @views p_tod = calc_p_tod(psi_tod, hwp_tod, IQU, theta_tod, phi_tod)
        
        @views @inbounds @simd for i in 1:size(pix_tod)[1]
            dp_map[pix_tod[i] , :] += p_tod[i] .* d_tod_true[i,:]
        end
    end
    @views @inbounds @threads for i in 1:size(dp_map)[1]
        @views IQU_out[i,:] = @views D_matrix[i,:,:] \ dp_map[i,:] 
    end
    return IQU_out
end

function calc_D_from_tod(ss::ScanningStrategy,　division::Int)　　# DのピクセルにハーフウェブプレートのTOD、ψのTOD
    section = Int(ss.duration / division)
    D = @views zeros(Float64, (npix, 3, 3,))
    @views @inbounds for j = 1:division
        
        BEGIN = (j - 1) * section
        END = j * section 
        @views pix_tod, psi_tod, time_array = get_pointing_pixels(ss, BEGIN, END);
        
        hwp_angle = @views zeros(size(pix_tod)[1]) 
        @views @inbounds @threads for i in 1:size(pix_tod)[1]
            hwp_angle[i] = 2π * ((ss.hwp_rpm * time_array[i] / 60) % (1))
        end
        
        @views @inbounds @simd for i in 1:size(pix_tod)[1]
            A = cos(4 * hwp_angle[i] - 2 * psi_tod[i])
            B = sin(4 * hwp_angle[i] - 2 * psi_tod[i])
            D[pix_tod[i],1,1] += 1.0
            D[pix_tod[i],1,2] +=  A
            D[pix_tod[i],1,3] +=  B
            D[pix_tod[i],2,3] += A * B
            D[pix_tod[i],3,3] += B^2 
            D[pix_tod[i],2,2] += A^2 
        end
    end
    @views D[:,3,1] = @views D[:,1,3] 
    @views D[:,3,2] = @views D[:,2,3] 
    @views D[:,2,1] = @views D[:,1,2] 
    return (D ./ 4)
end
println("関数定義ゾーン終わり")
# 関数定義ゾーン終わり


# クラス、変数定義ゾーン
SS_true = gen_ScanningStrategy(
    prec_rpm=period2rpm(prec_period),
    alpha=45.0, # [degree],
    nside=NSIDE,
    duration=duration_period , # [sec],
    sampling_rate=samp_rate , # [Hz],
    FP_theta=[0.0] , 
    FP_phi=[0.0],
    beta=50.0 ,
    spin_rpm=0.05,
    hwp_rpm=46,
    start_point="pole" 
)

println("#クラス、定義ゾーン終わり")

D = calc_D_from_tod(SS_true, division)　# D_matrix計算
IQU_input = hp.read_map("/home/cmb/yusuket/program/MapData/Nside512/lensed_r0_512_non_smooth.fits", [0,1,2], )
IQU_input  = hp.smoothing(IQU_input, np.deg2rad(0.5))'# input mapの呼び出し



# input map TQU で計算
IQU_estimate = calc_IQU_from_tod(SS_true, division, D, IQU_input)
println("end IQU")
np.savez("/group/cmb/litebird/usr/naoyadoi/data/IQU_data/takasesan" * "/interped_IQU" , IQU_estimate) 

# input map T=0 QUで計算

println("")
println("")