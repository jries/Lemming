package org.lemming.processors;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import org.lemming.data.XYFLocalization;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;

/**
 * @author Ronny Sczech
 *
 * @param <T> - pixel type
 * @param <F> - frame type
 */
public class NMSFinder<T extends RealType<T>, F extends Frame<T>> extends SingleInputSingleOutput<F, Localization> {

	private boolean hasMoreOutputs;
	private float cutoff;
	private int size;
	private long start;
	
	/**
	 * @param threshold - minimum threshold for a peak 
	 * @param size - size of the kernel
	 * 
	 */
	public NMSFinder(final float threshold, final int size){
		hasMoreOutputs = true;
		this.size = size;
		cutoff = threshold;
		start=System.currentTimeMillis();
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return hasMoreOutputs;
	}

	@Override
	public void process(F frame) {
		if (frame==null) return;
		process1(frame);
		if (frame.isLast()){
			long end = System.currentTimeMillis();
			System.out.println("Last frame finished:"+frame.getFrameNumber()+" in "+(end-start)+" ms");
			XYFLocalization lastloc = new XYFLocalization(frame.getFrameNumber(), 0, 0);
			lastloc.setLast(true);
			output.put(lastloc);
			hasMoreOutputs = false;
			stop();
			return;
		}
		if (frame.getFrameNumber() % 500 == 0)
			System.out.println("Frames finished:"+frame.getFrameNumber());
	}

	private void process1(F frame) {
		final RandomAccessibleInterval<T> interval = frame.getPixels();
		RandomAccess<T> ra = interval.randomAccess();
		
		int i,j,ii,jj,ll,kk;
		int mi,mj;
		boolean failed=false;
		int n_=size;
		long width_= interval.dimension(0);
		long height_ = interval.dimension(1);
	
		for(i=n_;i<=width_-1-n_;i+=n_+1){	// Loop over (n+1)x(n+1)
			for(j=n_;j<=height_-1-n_;j+=n_+1){
				mi = i;
				mj = j;
				for(ii=i;ii<=i+n_;ii++){	
					for(jj=j;jj<=j+n_;jj++){
						ra.setPosition(new int[]{ii,jj});
						T first = ra.get();
						ra.setPosition(new int[]{mi,mj});
						T second = ra.get();
						if (first.compareTo(second) > 0){	
							mi = ii;
							mj = jj;
						}
					}
				}
				failed = false;
				
				Outer:
				for(ll=mi-n_;ll<=mi+n_;ll++){	
					for(kk=mj-n_;kk<=mj+n_;kk++){
						if((ll<i || ll>i+n_) || (kk<j || kk>j+n_)){
							if(ll<width_ && ll>0 && kk<height_ && kk>0){
								ra.setPosition(new int[]{ll,kk});
								T first = ra.get();
								ra.setPosition(new int[]{mi,mj});
								T second = ra.get();
								if(first.compareTo(second) > 0){
									failed = true;
									break Outer;
								}
							}
						}
					}
				}
				if(!failed){
					ra.setPosition(new int[]{mi,mj});
					T first = ra.get();
					if(first.getRealFloat() > cutoff){
						output.put(new XYFLocalization(frame.getFrameNumber(), mi, mj));
					}
				}
			}			
		}	
	}
	
}
