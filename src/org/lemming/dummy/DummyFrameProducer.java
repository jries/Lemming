package org.lemming.dummy;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.queue.Store;
import org.lemming.interfaces.Source;

public class DummyFrameProducer implements Source<ImgLib2Frame<UnsignedShortType>> {

        int i = 0;
	
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
	public void beforeRun() {
 	}
	
        @Override
	public void afterRun() {
 	}
	
	@Override
	public ImgLib2Frame<UnsignedShortType> newOutput() {
                return new DummyFrame(i++);
	}
			
	@Override
	public boolean hasMoreOutputs() {
		return i < 100;
	}

}
