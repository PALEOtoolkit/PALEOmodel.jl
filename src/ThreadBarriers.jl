""" 
    Thread barriers (https://en.wikipedia.org/wiki/Barrier_(computer_science))

    See also https://github.com/JuliaConcurrent/SyncBarriers.jl
"""
module ThreadBarriers


"""
    wait_barrier(barrier::Nothing)

Null implementation of `wait_barrier` for single-threaded case.
"""
function wait_barrier(::Nothing)
    return nothing
end


"""
    ThreadBarrierAtomic

Thread synchronisation barrier for `Threads.nthreads()` Julia Threads. 
Uses a pair of Atomic variables to avoid the need for locks. Resets so can be reused.

# Example:

    barrier = ThreadBarrierAtomic("my barrier")    
    Threads.@threads for t in 1:Threads.nthreads()  
        # do stuff
        
        wait_barrier(barrier)  # blocks until all threads reach this point

        # do more stuff
    end
"""
mutable struct ThreadBarrierAtomic
    id::String
    NTHREADS::Int
    nbarrier::Threads.Atomic{Int}
    generation::Threads.Atomic{Int}

    ThreadBarrierAtomic(id) = new(id, Threads.nthreads(), Threads.Atomic{Int}(Threads.nthreads()), Threads.Atomic{Int}(0))   
end

"""
    wait_barrier(barrier::ThreadBarrierAtomic)

Busy wait until all `Threads.nthreads()` arrived at `barrier`
"""
function wait_barrier(barrier::ThreadBarrierAtomic)
   
    my_generation = barrier.generation[]
 
    if(Threads.atomic_sub!(barrier.nbarrier, 1) == 1) # atomic_sub! returns the old value       
        barrier.nbarrier[] = barrier.NTHREADS
        barrier.generation[] = my_generation + 1
    else       
        # See
        # https://github.com/JuliaLang/julia/issues/33097
        # https://github.com/JuliaLang/julia/pull/33092
        # GC.safepoint() is essential to avoid deadlocks
        # jl_cpu_pause is an optimisation (see implementation of SpinLock() in Julia base/locks-mt.jl)
        # Using a counter to reduce the rate of calls to GC.safepoint() is a guess at an optimisation
        i = 0
        while(barrier.generation[] == my_generation)
            ccall(:jl_cpu_pause, Cvoid, ())
            i += 1
            if (i % 1000000) == 0
                # see https://github.com/JuliaLang/julia/issues/33097
                # https://github.com/JuliaLang/julia/pull/33092
                GC.safepoint()
            end
        end
    end

    return nothing
end


"""
    ThreadBarrierCond

Thread synchronisation barrier for `Threads.nthreads()` Julia Threads using Condition variable
and associated lock. Resets so can be reused. 

NB: testing on Julia 1.6 suggests this is slow. 

# Example:

    barrier = ThreadBarrierCond("my barrier")    
    Threads.@threads for t in 1:Threads.nthreads()  
        # do stuff
        
        wait_barrier(barrier)  # blocks until all threads reach this point

        # do more stuff
    end

# Implementation:
Uses a condition variable (with associated lock) and a counter.
See eg <http://web.eecs.utk.edu/~huangj/cs360/360/notes/CondVar/lecture.html>

"""
mutable struct ThreadBarrierCond
    id::String
    # TODO fails if cond Type provided here ??
    cond #::Base.GenericCondition{ReentrantLock}
    ndone::Ref{Int}
    NTHREADS::Int
    debug::Bool

    ThreadBarrierCond(id; debug=false) = new(id, Threads.Condition(), Ref{Int}(0), Threads.nthreads(), debug)   
end

"""
    wait_barrier(barrier::ThreadBarrierCond)

Wait until all `Threads.nthreads()` arrived at `barrier`
"""
function wait_barrier(barrier::ThreadBarrierCond) #, cond)
    barrier.debug && println("Thread $(Threads.threadid()) about to acquire barrier '$(barrier.id)' lock")

    lock(barrier.cond)

    barrier.ndone[] += 1

    if barrier.ndone[] < barrier.NTHREADS
        barrier.debug && println("Thread $(Threads.threadid()) waiting for barrier '$(barrier.id)' ndone $(barrier.ndone[]) NTHREADS $(barrier.NTHREADS) ...")

        wait(barrier.cond) # releases lock, requires before returning
    else
        barrier.debug && println("Thread $(Threads.threadid()) about to notify barrier '$(barrier.id)' ...")

        barrier.ndone[] = 0 # reset barrier
        
        notify(barrier.cond)      
    end
    unlock(barrier.cond)

    barrier.debug && println("Thread $(Threads.threadid()) after barrier '$(barrier.id)' ...")

    return nothing
end

end # module
