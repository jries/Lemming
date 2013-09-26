package org.lemming.dummy;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Localizer;

public class DummyLocalizer implements Localizer {

	Store<Frame> in;
	Store<Localization> out;
	
	@Override
	public void setInput(Store<Frame> s) {
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
		public double getY() { return y; };
	}
	
	@Override
	public void run() {
		while(true) {
			Frame f = in.get();
			
			out.put(new DummyLocalization(f.getFrameNumber(), 0));
			out.put(new DummyLocalization(0, f.getFrameNumber()));
		}
	}
}
