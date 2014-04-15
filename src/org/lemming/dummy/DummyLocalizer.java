package org.lemming.dummy;

import java.util.AbstractList;
import java.util.ArrayList;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.queue.Store;
import org.lemming.interfaces.ImageLocalizer;

public class DummyLocalizer<T, F extends Frame<T>> implements ImageLocalizer<T,F> {
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
	public AbstractList<Localization> process(F f) {
                ArrayList<Localization> result = new ArrayList<Localization>();
                result.add(new DummyLocalization(f.getFrameNumber(), 0));
                result.add(new DummyLocalization(0, f.getFrameNumber()));
                return result;
	}

}
