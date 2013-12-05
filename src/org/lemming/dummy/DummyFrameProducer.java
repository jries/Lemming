package org.lemming.dummy;

import org.lemming.data.Frame;
import org.lemming.data.Store;
import org.lemming.interfaces.Source;

public class DummyFrameProducer implements Source<Frame> {

	Store<Frame> output;
	
	@Override
	public void setOutput(Store<Frame> store) {
		output = store;
	}
	
	class DummyFrame implements Frame {
		long ID;
		
		DummyFrame(long ID) {
			this.ID = ID;
		}
		
		public long getFrameNumber() {
			return ID;
		}

		@Override
		public Object getPixels() {
			return null;
		}

		@Override
		public int getWidth() {
			return 0;
		}

		@Override
		public int getHeight() {
			return 0;
		}
	}
	
	@Override
	public void run() {
		for(int i=0; i<100; i++)
			output.put(new DummyFrame(i));
		
		hasMore = false;
	}

	boolean hasMore = true;
			
	@Override
	public boolean hasMoreOutputs() {
		return hasMore;
	}

}
