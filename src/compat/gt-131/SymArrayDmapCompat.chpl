
module SymArrayDmapCompat
{
    use ChplConfig;

    /*
     Available domain maps.
     */
    enum Dmap {defaultRectangular, blockDist};

    private param defaultDmap = if CHPL_COMM == "none" then Dmap.defaultRectangular
                                                       else Dmap.blockDist;
    /*
    How domains/arrays are distributed. Defaults to :enum:`Dmap.defaultRectangular` if
    :param:`CHPL_COMM=none`, otherwise defaults to :enum:`Dmap.blockDist`.
    */
    config param MyDmap:Dmap = defaultDmap;

    public use BlockDist;
    use GPUUnifiedDist;

    /* 
    Makes a domain distributed according to :param:`MyDmap`.

    :arg size: size of domain
    :type size: int
    */
    proc makeDistDom(size:int, param GPU:bool = false) where GPU == true {
        select MyDmap
        {
            when Dmap.defaultRectangular {
                return {0..#size} dmapped gpuUnifiedDist(boundingBox={0..#size});
            }
            when Dmap.blockDist {
                if size > 0 {
                    return {0..#size} dmapped gpuUnifiedDist(boundingBox={0..#size});
                }
                // fix the annoyance about boundingBox being enpty
                else {return {0..#0} dmapped gpuUnifiedDist(boundingBox={0..0});}
            }
            otherwise {
                halt("Unsupported distribution " + MyDmap:string);
            }
        }
    }

    proc makeDistDom(size:int, param GPU:bool = false) {
        select MyDmap
        {
            when Dmap.defaultRectangular {
                return {0..#size};
            }
            when Dmap.blockDist {
                if size > 0 {
                    return {0..#size} dmapped blockDist(boundingBox={0..#size});
                }
                // fix the annoyance about boundingBox being enpty
                else {return {0..#0} dmapped blockDist(boundingBox={0..0});}
            }
            otherwise {
                halt("Unsupported distribution " + MyDmap:string);
            }
        }
    }
    
    /* 
    Makes an array of specified type over a distributed domain

    :arg size: size of the domain
    :type size: int 

    :arg etype: desired type of array
    :type etype: type

    :returns: [] ?etype
    */
    proc makeDistArray(size:int, type etype, param GPU:bool = false) throws {
      var dom = makeDistDom(size, GPU);
      return dom.tryCreateArray(etype);
    }

    proc makeDistArray(in a: [?D] ?etype, param GPU:bool = false) throws
      where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular() {
        var res = makeDistArray(D.size, etype, GPU);
        res = a;
        return res;
    }
    
    proc makeDistArray(in a: [?D] ?etype) throws {
      var res = D.tryCreateArray(etype);
      res = a;
      return res;
    }

    proc makeDistArray(D: domain(?), type etype) throws {
      var res = D.tryCreateArray(etype);
      return res;
    }

    proc makeDistArray(D: domain(?), initExpr: ?t) throws {
      return D.tryCreateArray(t, initExpr);
    }

    /* 
    Returns the type of the distributed domain

    :arg size: size of domain
    :type size: int

    :returns: type
    */
    proc makeDistDomType(size: int, GPU:bool = false) type {
        return makeDistDom(size, GPU).type;
    }

}
