
# perform significance test with 5 surrogates
d18O_wwz_sig = d18O_wwz.signif_test(number=5)

# plot the results
_ = d18O_wwz_sig.plot(xlabel="Period [kyrs]")