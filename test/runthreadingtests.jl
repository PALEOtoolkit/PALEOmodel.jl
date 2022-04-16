import PALEOmodel
using Test


function test_barrier(barrier; nloops=1000, debug=false)    
    
    println("before loop Threads.threadid() ", Threads.threadid())

    testarray = zeros(Int, Threads.nthreads())
    Threads.@threads for t in 1:Threads.nthreads()
        println("Start Threads.threadid() ", Threads.threadid())

        for b = 1:nloops
            debug && println("Thread $(Threads.threadid()) waiting for barrier $b ...")
            
            PALEOmodel.ThreadBarriers.wait_barrier(barrier)

            debug && println("Thread $(Threads.threadid()) after barrier 1 $b ...")
            testarray[Threads.threadid()] = b

            PALEOmodel.ThreadBarriers.wait_barrier(barrier)

            # check that all threads:
            # (i) have updated testarray before being allowed to pass the second barrier
            # (ii) are prevented by the first barrier from updating testarray again on the next loop iteration
            if Threads.threadid() == 1
                @test all(testarray .== b)
            end

            debug && println("Thread $(Threads.threadid()) after barrier 2 $b ...")
        end


    end

    println("after loop Threads.threadid() ", Threads.threadid())
    return nothing
end


@testset "threadbarriers" begin

@testset "barriercond" begin
    test_barrier(PALEOmodel.ThreadBarriers.ThreadBarrierCond("test"))
end

@testset "barrieratomic" begin
    test_barrier(PALEOmodel.ThreadBarriers.ThreadBarrierAtomic("test"))
end

end