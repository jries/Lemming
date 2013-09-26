package org.lemming.dummy;

import org.lemming.data.Frame;
import org.lemming.data.Store;
import org.lemming.interfaces.Input;

public class DummyFrameProducer implements Input {

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
	}
	
	@Override
	public void run() {
		for(int i=0; i<100; i++)
			output.put(new DummyFrame(i));
	}

}
