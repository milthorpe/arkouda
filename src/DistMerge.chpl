module DistMerge {
  /*
   Provides a distributed merge sort of a partially-sorted distributed array, 
   where the data in each locale's local slice are already sorted
  */
    use Time;

    proc mergeSortedChunks(src, dst) {
        var buffer = new DoubleBuffer(src, dst);
        mergePartitions(buffer, src.domain.targetLocales());
    }

    class DoubleBuffer {
        var primary, secondary;
        proc flip() {
            primary <=> secondary;
        }
    }

    proc selectPivot(A: [], targetLocales: [] locale) {
        var startLH = A.localSubdomain(targetLocales.first).first;
        var startRH = A.localSubdomain(targetLocales[targetLocales.size/2]).first;
        var endLH = A.localSubdomain(targetLocales[targetLocales.size/2 - 1]).last + 1;
        //writeln("startLH = ", startLH, " startRH = ", startRH, " endLH = ", endLH);
        var lo = startLH;
        var hi = endLH;
        while lo < hi {
            var mid = hi - (hi - lo) / 2;
            //writeln("lo = ", lo, " hi = ", hi, " mid = ", mid);
            //writeln("A[", (endLH - mid + startLH), "] = ", A[endLH - mid + startLH], " A[", (mid-1),"] = ", A[mid-1]);
            if A[endLH - mid + startLH] <= A[mid-1] then // TODO fix for padded A
            hi = mid - 1;
            else
            lo = mid;
        }
        return lo - startLH;
    }

    proc mergePartitions(buffer: DoubleBuffer(?), targetLocales: [] locale) {
        if targetLocales.size > 2 then
            forall i in 0..1 {
                //writeln(dateTime.now(), " before need to merge targetLocales ", i * (targetLocales.size / 2), " and ", i * (targetLocales.size / 2) + 1, " of ", targetLocales.size);
                mergePartitions(buffer, [targetLocales[i * (targetLocales.size / 2)], targetLocales[i * (targetLocales.size / 2) + 1]]);
            }

        //writeln(dateTime.now(), " merging ", targetLocales);
        var pivot = selectPivot(buffer.primary, targetLocales);
        if pivot > 0 {
            var localesToMerge = swapPartitions(buffer, targetLocales, pivot);
            //writeln(dateTime.now(), " before merge sorted");
            mergeSortedKeys(buffer.primary, buffer.secondary, localesToMerge, pivot);
            buffer.flip();
        }

        if targetLocales.size > 2 then
            forall i in 0..1 {
            //writeln(dateTime.now(), " after need to merge targetLocales ", i * (targetLocales.size / 2), " and ", i * (targetLocales.size / 2) + 1, " of ", targetLocales.size);
            mergePartitions(buffer, [targetLocales[i * (targetLocales.size / 2)], targetLocales[i * (targetLocales.size / 2) + 1]]);
        }
    }

    proc swapPartitions(buffer: DoubleBuffer(?), targetLocales: [] locale, in pivot: int) {
        var partition_size = buffer.primary.size / buffer.primary.domain.targetLocales().size;
        var devices_to_swap = pivot / partition_size;
        //writeln(dateTime.now(), " pivot = ", pivot, " partition_size = ", partition_size);

        if (pivot == partition_size * (targetLocales.size / 2)) {
            devices_to_swap -= 1;
            pivot = partition_size;
        } else {
            pivot %= partition_size;
        }
        //writeln("targetLocales = ", targetLocales, " pivot = ", pivot, " devices_to_swap = ", devices_to_swap);

        var devices_to_merge: [0..1] (locale, int);

        forall i in 0..devices_to_swap {
            const left_device = targetLocales[targetLocales.size / 2 - i - 1];
            const right_device = targetLocales[targetLocales.size / 2 + i];
            const num_elements = if (i == devices_to_swap) then pivot else partition_size;

            const left_start = buffer.primary.localSubdomain(left_device).first + partition_size - num_elements;
            const right_start = buffer.primary.localSubdomain(right_device).first;
            //writeln(dateTime.now(), " partition_size ", partition_size, " left_device ", left_device, " right_device ", right_device, " num_elements ", num_elements, " left_start ", left_start, " right_start ", right_start);
            //writeln("left  ", buffer.primary[left_start..#num_elements]);
            //writeln("right ", buffer.primary[right_start..#num_elements]);
            cobegin {
                on right_device do buffer.secondary[right_start..#num_elements] = buffer.primary[left_start..#num_elements];
                on left_device do buffer.secondary[left_start..#num_elements] = buffer.primary[right_start..#num_elements];
            }

            //writeln(dateTime.now(), " before copy");

            if (i == devices_to_swap) {
                devices_to_merge[0] = (left_device,left_start);
                devices_to_merge[1] = (right_device,right_start+num_elements);
                forall j in 0..1 do on devices_to_merge[j] {
                    var device_domain = buffer.primary.localSubdomain(devices_to_merge[j][0]);
                    var slice_to_copy = device_domain.first + j * pivot .. #partition_size - pivot;
                    buffer.secondary[slice_to_copy] = buffer.primary[slice_to_copy];
                }
            }
        }
        buffer.flip();

        //writeln("now A = ", buffer.primary);

        /*
        forall j in 0..1 {
            const device = devices[devices_to_merge[j]];
            const start = A.localSubdomain(Locales[device]).first;
            A[start+j*pivot..start+partition_size-pivot] = secondaryA[start+j*pivot..start+partition_size-pivot];
        }
        // now flip buffers
        */
        /*
        var end0 = A.localSubdomain(targetLocales[0]).last;
        var start0 = end0 - pivot + 1;
        var start1 = A.localSubdomain(targetLocales[1]).first;
        var end1 = start1 + pivot - 1;
        writeln("swapping ", start0, "..", end0, " and ", start1, "..", end1);
        var tmp = A[start1..end1];
        A[start1..end1] = A[start0..end0];
        A[start0..end0] = tmp;
        */
        return devices_to_merge;
    }
}
