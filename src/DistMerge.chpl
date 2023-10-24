module DistMerge {
  /*
   Provides a distributed merge sort of a partially-sorted distributed array, 
   where the data in each locale's local slice are already sorted
  */
    use Time;
    use CommPrimitives;
    use CTypes;

    proc mergeSortedChunks(ref src) {
        var tmp: [src.domain] src.eltType = noinit;
        mergePartitions(src, tmp, src.domain.targetLocales());
    }

    proc selectPivot(A: [], targetLocales: [] locale) {
        var endLH = A.localSubdomain(targetLocales[targetLocales.size/2 - 1]).last;
        var startRH = A.localSubdomain(targetLocales[targetLocales.size/2]).first;
        var partitionSize = A.localSubdomain(targetLocales.first).size;
        //writeln("endLH = ", endLH, " startRH = ", startRH);
        var lo = 0;
        var hi = partitionSize * targetLocales.size/2;
        while lo < hi {
            var mid = hi - (hi - lo) / 2;
            //writeln("lo = ", lo, " hi = ", hi, " mid = ", mid);
            //writeln("A[", (startRH + mid-1), "] = ", A[startRH + mid-1], " A[", (endLH-mid+1),"] = ", A[endLH-mid+1]);
            if A[endLH-mid+1] <= A[startRH + mid-1] then // TODO fix for padded A
                hi = mid - 1;
            else
                lo = mid;
        }
        return lo;
    }

    proc swapPartitions(ref src, ref tmp, const targetLocales: [] locale, in pivot: int) {
        var partitionSize = src.size / src.domain.targetLocales().size;
        var localesToSwap = pivot / partitionSize;
        //writeln(dateTime.now(), " pivot = ", pivot, " partitionSize = ", partitionSize);

        if (pivot == partitionSize * (targetLocales.size / 2)) {
            localesToSwap -= 1;
            pivot = partitionSize;
        } else {
            pivot %= partitionSize;
        }
        //writeln("targetLocales = ", targetLocales, " pivot = ", pivot, " localesToSwap = ", localesToSwap);

        var localesToMerge: [0..1] (locale, int);

        coforall i in 0..localesToSwap with (const targetLocales, ref localesToMerge) {
            const leftLocale = targetLocales[targetLocales.size / 2 - i - 1];
            const rightLocale = targetLocales[targetLocales.size / 2 + i];
            const numElements = if (i == localesToSwap) then pivot else partitionSize;

            const leftStart = src.localSubdomain(leftLocale).first + partitionSize - numElements;
            const rightStart = src.localSubdomain(rightLocale).first;

            //writeln(dateTime.now(), " partitionSize ", partitionSize, " leftLocale ", leftLocale, " rightLocale ", rightLocale, " numElements ", numElements, " leftStart ", leftStart, " rightStart ", rightStart);
            //writeln("left  ", src[leftStart..#numElements]);
            //writeln("right ", src[rightStart..#numElements]);

            var rightSrc: c_ptr(src.eltType);
            on rightLocale do rightSrc = c_ptrTo(src[rightStart]);
            var leftSrc: c_ptr(src.eltType);
            on leftLocale do leftSrc = c_ptrTo(src[leftStart]);
            cobegin {
                // TODO would like to be able to use slices as follows, but they generate too much extra communication
                //on leftLocale do tmp.localSlice(leftStart..#numElements) = src[rightStart..#numElements];
                //on rightLocale do tmp.localSlice(rightStart..#numElements) = src[leftStart..#numElements];
                on leftLocale do GET(c_ptrTo(tmp[leftStart]), rightSrc, rightLocale.id, numElements*c_sizeof(src.eltType));
                on rightLocale do GET(c_ptrTo(tmp[rightStart]), leftSrc, leftLocale.id, numElements*c_sizeof(src.eltType));
            }

            //writeln(dateTime.now(), " before copy");

            if (i == localesToSwap) {
                localesToMerge[0] = (leftLocale,leftStart);
                localesToMerge[1] = (rightLocale,rightStart+numElements);

                coforall j in 0..1 do on localesToMerge[j](0) {
                    var localSub = src.localSubdomain(here);
                    var sliceToCopy = (localSub.first + j * pivot) .. #(partitionSize - pivot);
                    tmp.localSlice(sliceToCopy) = src.localSlice(sliceToCopy);
                }
            } else {
                // these locales won't be merged, so copy elements back to src
                coforall loc in {leftLocale, rightLocale} do on loc {
                    const localSub = src.localSubdomain();
                    src.localSlice(localSub) = tmp.localSlice(localSub);
                }
            }
            //writeln(dateTime.now(), " after copy");
        }

        return localesToMerge;
    }

    proc mergePartitions(ref src, ref tmp, const targetLocales: [] locale) {
        if targetLocales.size > 2 then
            coforall i in 0..1 {
                var localesSubset: [0..#targetLocales.size / 2] locale;
                localesSubset = targetLocales((i * (targetLocales.size / 2)) .. #(targetLocales.size / 2));
                mergePartitions(src, tmp, localesSubset);
            }

        //writeln(dateTime.now(), " merging ", targetLocales);
        //writeln(dateTime.now(), " src ", src);
        var pivot = selectPivot(src, targetLocales);
        //writeln(dateTime.now(), " pivot ", targetLocales, " = ", pivot);
        if pivot > 0 {
            var localesToMerge = swapPartitions(src, tmp, targetLocales, pivot); // src->tmp
            //writeln(dateTime.now(), " swapped ", targetLocales);
            //writeln(dateTime.now(), " ", tmp);
            coforall (loc, cut) in localesToMerge do on loc {
                mergeSortedKeysAtCut(src, tmp, cut); // tmp->src
            }
            //writeln(dateTime.now(), " merged ", targetLocales);
            //writeln(dateTime.now(), " ", src);
        }

        if targetLocales.size > 2 then
            coforall i in 0..1 {
                var localesSubset: [0..#targetLocales.size / 2] locale;
                localesSubset = targetLocales((i * (targetLocales.size / 2)) .. #(targetLocales.size / 2));
                mergePartitions(src, tmp, localesSubset);
            }
    }
}
