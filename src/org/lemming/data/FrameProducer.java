package org.lemming.data;

public class FrameProducer implements Frame {

	long currentFrameNumber = 0L;
	Object object;
	
	public FrameProducer(Object object) {
		this.object = object;
		currentFrameNumber++;
	}
		
	@Override
	public long getFrameNumber() {		
		return currentFrameNumber;
	}

	@Override
	public Object getPixels() {
		return object;
	}

}
