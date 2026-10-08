# Deterministic, asymmetric multi-scale texture, without a random-library dependency.
function fixture(h=160,w=192)
    a = fill(0.15f0,h,w)
    for k in 1:70
        y = 12 + mod(37k, h-24)
        x = 12 + mod(61k, w-24)
        sy = 1.8 + mod(7k,15)/3
        sx = 1.8 + mod(11k,15)/3
        amp = 0.25 + mod(13k,7)/14
        for j in 1:w, i in 1:h
            a[i,j] += Float32(amp*exp(-((i-y)^2/sy^2+(j-x)^2/sx^2)/2))
        end
    end
    clamp.(a,0f0,1f0)
end
