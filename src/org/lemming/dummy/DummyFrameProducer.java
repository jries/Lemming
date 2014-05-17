package org.lemming.dummy;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Store;
import org.lemming.interfaces.Parameter;
import org.lemming.interfaces.Source;

public class DummyFrameProducer implements Source<ImgLib2Frame<UnsignedShortType>> {

	@Parameter(readOnly = true, optional = false)
	public long noFrames = 0;
	
	Store<ImgLib2Frame<UnsignedShortType>> output;
	
	public DummyFrameProducer() {
		this(100);
	}
	
	public DummyFrameProducer(long noFrames) {
		this.noFrames = noFrames;
	}
	
	@Override
	public void setOutput(Store<ImgLib2Frame<UnsignedShortType>> store) {
		output = store;
	}
	
	class DummyFrame extends ImgLib2Frame<UnsignedShortType> {
		long ID;
		
		DummyFrame(long ID) {
			super(ID, 256, 256, ArrayImgs.unsignedShorts(new long[]{256,256}));
			
			this.ID = ID;
		}
		
		public long getFrameNumber() {
			return ID;
		}

		@Override
		public RandomAccessibleInterval<UnsignedShortType> getPixels() {
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
		for(int i=0; i<noFrames; i++)
			output.put(new DummyFrame(i));
		
		hasMore = false;
	}

	boolean hasMore = true;
			
	@Override
	public boolean hasMoreOutputs() {
		return hasMore;
	}

}
