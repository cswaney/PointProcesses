using Distributions

function assert_nonnegative(x, msg)
    if any(x .< 0)
        @error msg
    end
end

function assert_probability(x, msg)
    if any(x .> 1) || any(x .< 0)
        @error msg
    end
end

function assert_valid_intensity(h::Distribution)
    supp = support(h)
    if (typeof(supp) != RealInterval)
        @error "Support of `h` must be continuous"
    end
    if (supp.lb < 0)
        @error "Support of `h` must be non-negative"
    end
end

function assert_valid_intensity(h::UnivariateLinear)
    supp = support(h)
    if (supp.lb < 0)
        @error "Support of `h` must be non-negative"
    end
    if !isnonnegative(h)
        @error "Range of `h` must be non-negative"
    end
end
