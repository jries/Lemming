package org.lemming.dummy;

//import java.util.concurrent.locks.Lock;
//import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.lemming.interfaces.Frame;
import org.lemming.interfaces.ImageLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class DummyLocalizer<T, F extends Frame<T>> implements ImageLocalizer<T,F> {

	/**
	 * input
	 */
	public Store<F> in;
	/**
	 * output
	 */
	public Store<Localization> out;
	private boolean hasMoreOutputs = true;
	
	/**
	 * 
	 */
	public DummyLocalizer() {
		super();
	}
	
	@Override
	public void setInput(Store<F> s) {
		in = s;
	}

	@Override
	public void setOutput(Store<Localization> s) {
		out = s;
	}
	
	static class DummyLocalization implements Localization {
		static long CUR_ID = 0;
		long ID;
		double x,y;

		public DummyLocalization(double x, double y) {
			ID = CUR_ID++;
			this.x = x;
			this.y = y;
		}
		
		@Override
		public long getID() { return ID; }
		
		@Override
		public double getX() { return x; }
		
		@Override
		public double getY() { return y; }

		@Override
		public boolean isLast() {
			return false;
		}

		public void setLast(boolean isLast) {
		}

	}
	
	@Override
	public void run() {
		Frame<T> f;
		do {
			f = in.get();
			//System.out.println("loop:"+f.getFrameNumber());
			out.put(new DummyLocalization(f.getFrameNumber(), 0));
			out.put(new DummyLocalization(0, f.getFrameNumber()));			
		} while(!f.isLast());
		hasMoreOutputs=false;			
		
		System.out.println("*** DummyLocalizer ****");
	}

	@Override
	public boolean hasMoreOutputs() {
		return hasMoreOutputs;
	}


}
