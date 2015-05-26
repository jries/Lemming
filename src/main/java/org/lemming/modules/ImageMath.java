package org.lemming.modules;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

public class ImageMath<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	public enum operators {
		ADDITION, SUBSTRACTION, MULTIPLICATION, DIVISION
	}

	private operators operator;
	private Pair<String,String> inputKeys;
	private int counter;
	private long start;
	private Store output;
	
	public ImageMath(Pair<String,String> ins){
		inputKeys = ins;
	}
	
	public void setOperator(operators op){
		operator = op;
	}
	
	@Override
	protected void beforeRun(){ // check for equal number in the two input stores
		output = outputs.values().iterator().next();
		iterator = inputKeys.getB();
		int length = 0;
		boolean loop = true;
		while(loop){
			for ( String key : inputs.keySet()){
				if (length == inputs.get(key).getLength())
					loop = false;
				length = inputs.get(key).getLength();
			}
			pause(10);
		}
		System.out.println("Image Math - Input ready");
		start = System.currentTimeMillis();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void process(Element data) {
		F frameB = (F) data;
		if (frameB==null){ 
			return;
		}
		F frameA = (F) inputs.get(inputKeys.getA());
		if (frameA==null){ 
			inputs.get(inputKeys.getB()).put(frameB);
			return;
		}
		
		// if no match put it back to inputs
		if (frameA.getFrameNumber() != frameB.getFrameNumber()){
			inputs.get(inputKeys.getB()).put(frameB);
			inputs.get(inputKeys.getA()).put(frameA);
			return;
		}		
		
		Pair<F,F> framePair= new ValuePair<>(frameA,frameB);
		
		if (frameA.isLast()){ // make the poison pill
			ImgLib2Frame<T> lastFrame = process1(framePair);
			lastFrame.setLast(true);
			output.put(lastFrame);
			cancel();
			counter++;
			return;
		}

		output.put(process1(framePair));
		counter++;		
		
		//if (counter % 100 == 0)
		//	System.out.println("Frames calculated:" + counter);
		
	}

	private ImgLib2Frame<T> process1(Pair<F, F> framePair) {
		
		RandomAccessibleInterval<T> intervalA = framePair.getA().getPixels();
		RandomAccessibleInterval<T> intervalB = framePair.getB().getPixels();
		
		Cursor<T> cursorA = Views.flatIterable(intervalA).cursor();
		Cursor<T> cursorB = Views.flatIterable(intervalB).cursor();
		
		switch (operator){
		case ADDITION:			
			while ( cursorA.hasNext()){
	            cursorA.fwd();  cursorB.fwd(); // move both cursors forward by one pixel
	            cursorA.get().add(cursorB.get());
	        }			
			break;
		case SUBSTRACTION:		
			while ( cursorA.hasNext()){
	            cursorA.fwd();  cursorB.fwd(); // move both cursors forward by one pixel
	            cursorA.get().sub(cursorB.get());
	        }
			break;
		case MULTIPLICATION:
			while ( cursorA.hasNext()){
	            cursorA.fwd();  cursorB.fwd(); // move both cursors forward by one pixel
	            cursorA.get().mul(cursorB.get());
	        }
			break;
		case DIVISION:
			while ( cursorA.hasNext()){
	            cursorA.fwd();  cursorB.fwd(); // move both cursors forward by one pixel
	            cursorA.get().div(cursorB.get());
	        }
			break;
		default:
		}
		
		return new ImgLib2Frame<>(framePair.getA().getFrameNumber(), framePair.getA().getWidth(), framePair.getA().getHeight(), intervalA);
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Math done with " + counter + " frames in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
