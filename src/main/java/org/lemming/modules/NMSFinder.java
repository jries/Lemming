package org.lemming.modules;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.MultiRunModule;

public class NMSFinder<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {
	
	private int size;
	private double threshold;
	private long start;
	private int counter;
	private Store output;

	public NMSFinder(final double threshold, final int size) {
		this.threshold = threshold;
		this.size = size;
	}

	@Override
	protected void beforeRun() {
		// for this module there should be only one key
		output = outputs.values().iterator().next(); 
		start = System.currentTimeMillis();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {
		F frame = (F) data;
		if (frame == null)
			return null;

		process1(frame);
		if (frame.isLast()) { // make the poison pill
			cancel();
			Localization lastloc = new Localization(frame.getFrameNumber(), -1, -1);
			lastloc.setLast(true);
			output.put(lastloc);
			return null;
		}
		//if (frame.getFrameNumber() % 500 == 0)
		//	System.out.println("Frames finished:" + frame.getFrameNumber());
		return null;
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
					if(first.getRealFloat() > threshold){
						output.put(new Localization(frame.getFrameNumber(), mi, mj));
						counter++;
					}
				}
			}			
		}	
	}
	
	@Override
	protected void afterRun() {
		System.out.println("NMSFinder found "
				+ counter + " peaks in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
