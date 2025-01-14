
function f!(resid, x, userdata=nothing)
    for i in eachindex(x)
        resid[i] = sin(x[i]) + x[i]^3
    end
end

x = ones(5)

kin = PALEOmodel.Kinsol.kin_create(
                            f!, x,
                            linear_solver = :FGMRES, 
                            krylov_dim=20)

y, kinstats =  PALEOmodel.Kinsol.kin_solve(
                            kin, x,
                            fnormtol=1e-8)

@test isapprox(y, zeros(5), atol=2e-8)
@info "FGMRES: kinstats = $kinstats"

function psetup(u, uscale, 
                fval, fscale,
                data)

    # println("psetup data=$data")
    return 0
end

function psolve(u, uscale, 
                fval, fscale,
                v, data)

    # println("psolve data=$data v=$v")                
    v .= v

    return 0

end


kin = PALEOmodel.Kinsol.kin_create(
                            f!, x,
                            linear_solver = :FGMRES, 
                            krylov_dim=20, 
                            psetupfun=psetup,
                            psolvefun=psolve,
                            userdata="hello")

y, kinstats =  PALEOmodel.Kinsol.kin_solve(
                            kin, x,
                            fnormtol=1e-8)

@test isapprox(y, zeros(5), atol=2e-8)

@info "FGMRES identity prec: kinstats = $kinstats"

function jv(v, Jv, 
        u, new_u,
        data)

    # @Infiltrator.infiltrate
    # println("jv data=$data v=$v")   

    r1 = similar(v)
    r2 = similar(v)

    sigma = 1e-6
    f!(r2, u + sigma.*v, data)
    f!(r1, u, data)

    Jv .= (r2 .- r1)./sigma

    # @Infiltrator.infiltrate

    return 0  # Success
end


kin = PALEOmodel.Kinsol.kin_create(
                        f!, x,
                        linear_solver = :FGMRES, 
                        krylov_dim=20, 
                        # psetupfun=psetup,
                        psolvefun=psolve,
                        jvfun = jv,
                        userdata="hello")

y, kinstats =  PALEOmodel.Kinsol.kin_solve(
                            kin, x,
                            fnormtol=1e-8)
    
@test isapprox(y, zeros(5), atol=2e-8)

# test reuse of kin instance
y, kinstats =  PALEOmodel.Kinsol.kin_solve(
                        kin, x,
                        fnormtol=1e-8,
                        print_level = 1,
                        mxiter=10)

@test isapprox(y, zeros(5), atol=2e-8)

@info "FGMRES identity prec jv: kinstats = $kinstats"