module Histogram
{
    use ServerConfig;
    
    use ArkoudaTimeCompat as Time;
    use Math only;

    use PrivateDist;
    use SymArrayDmapCompat;
    use Logging;
    use Reflection;
    use GPUIterator;
    use CUBHistogram;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const hgLogger = new Logger(logLevel, logChannel);

    /*
    Takes the data in array a, creates an atomic histogram in parallel, 
    and copies the result of the histogram operation into a distributed int array

    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramGlobalAtomic(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) throws {

        var hD = makeDistDom(bins);
        var atomicHist: [hD] atomic int;
        
        // count into atomic histogram
        forall v in a {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            if (vBin < 0) | (vBin > (bins-1)) {
                try! hgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),"OOB");
            }
            atomicHist[vBin].add(1);
        }
        
        var hist = makeDistArray(bins,int);
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomicHist)] e = ae.read();
        try! hgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "hist = %?".doFormat(hist));

        return hist;
    }

    /*
    Takes the data in array a, creates an atomic histogram in each locale,
    + reduces each locale's histogram computations into a distributed int array

    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramLocalAtomic(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) throws {

        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#bins] atomic int;

        // count into per-locale private atomic histogram
        forall v in a {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            atomicHist[here.id][vBin].add(1);
        }

        // +reduce across per-locale histograms to get counts
        var lHist: [0..#bins] int;
        forall i in PrivateSpace with (+ reduce lHist) do
          lHist reduce= atomicHist[i].read();

        var hist = makeDistArray(bins,int);        
        hist = lHist;
        return hist;
    }
    

    /*
    Iterates in parallel over all values of a, histogramming into a new array as each value is processed. 
    This new array is returned as the histogram.
    
    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramReduceIntent(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) throws {

        var gHist: [0..#bins] int;
        
        // count into per-task/per-locale histogram and then reduce as tasks complete
        forall v in a with (+ reduce gHist) {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            gHist[vBin] += 1;
        }

        var hist = makeDistArray(bins,int);        
        hist = gHist;
        return hist;
    }


}
