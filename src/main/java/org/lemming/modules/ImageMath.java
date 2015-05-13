package org.lemming.modules;

import java.util.Map;

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

public class ImageMath<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	public enum operators {
		ADDITION, SUBSTRACTION, MULTIPLICATION, DIVISION
	}

	private operators operator;
	private String outputKey;
	private Pair<String,String> inputKeys;
	private int counter;
	private long start;
	
	public ImageMath(Pair<String,String> ins, String out){
		inputKeys = ins;
		outputKey = out;
	}
	
	public void setOperator(operators op){
		operator = op;
	}


	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		F frameA = (F) data.get(inputKeys.getA());
		if (frameA==null) return;
		F frameB = (F) data.get(inputKeys.getB());
		if (frameB==null) return;
		
		// if no match put it back to inputs
		if (frameA.getFrameNumber() != frameB.getFrameNumber()){
			inputs.get(inputKeys.getB()).put(frameB);
			inputs.get(inputKeys.getA()).put(frameA);
			return;
		}		
		
		Pair<F,F> framePair= new ValuePair<>(frameA,frameB);
		
		if (frameA.isLast()){ // make the poison pill
			if (inputs.get(inputKeys.getA()).isEmpty()){
				process1(framePair);
				cancel();
				return;
			}
			inputs.get(inputKeys.getA()).put(frameA); //if queue is not empty put back to the end
			inputs.get(inputKeys.getB()).put(frameB);
			return;
		}
		
		ImgLib2Frame<T> out = process1(framePair);
		outputs.get(outputKey).put(out);
		counter++;
		//if (counter % 100 == 0)
		//	System.out.println("Frames calculated:"+counter);
		
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
	protected void beforeRun(){
		start = System.currentTimeMillis();
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Math done " + counter +" in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
