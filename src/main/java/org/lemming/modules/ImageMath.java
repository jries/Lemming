package org.lemming.modules;

import java.util.Iterator;
import java.util.NoSuchElementException;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;

public class ImageMath<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	public enum operators {
		ADDITION, SUBSTRACTION, MULTIPLICATION, DIVISION
	}

	private operators operator;
	private int counter;
	private long start;
	private Store inputA;
	private Store inputB;
	private Store output;
	
	public ImageMath(){
	}
	
	public void setOperator(operators op){
		operator = op;
	}
	
	@Override
	protected void beforeRun(){ 
		Iterator<Integer> it = inputs.keySet().iterator();
		try {
			iterator = it.next();							// first input
			inputB = inputs.get(iterator);
			inputA = inputs.get(it.next());
			output = outputs.values().iterator().next(); 	// output
		} catch (NoSuchElementException | NullPointerException ex){
			System.err.println("Input provided not correct!");
			Thread.currentThread().interrupt();
		}
	
		int length = 0;
		boolean loop = true;
		while(loop){									// check for equal number in the two input stores
			for ( Integer key : inputs.keySet()){
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
	public Element process(Element data) {
		F frameB = (F) data;
		if (frameB==null){ 
			return null;
		}
		F frameA = (F) inputA.get();
		if (frameA==null){ 
			inputB.put(frameB);
			return null;
		}
		
		// if no match put it back to inputs
		if (frameA.getFrameNumber() != frameB.getFrameNumber()){
			inputB.put(frameB);
			inputA.put(frameA);
			return null;
		}		
		
		Pair<F,F> framePair= new ValuePair<>(frameA,frameB);
		
		if (frameA.isLast()){ // make the poison pill
			ImgLib2Frame<T> lastFrame = process1(framePair);
			lastFrame.setLast(true);
			output.put(lastFrame);
			cancel();
			counter++;
			return null;
		}

		output.put(process1(framePair));
		counter++;		
		
		//if (counter % 100 == 0)
		//	System.out.println("Frames calculated:" + counter);
		return null;
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

	@Override
	public boolean check() {
		return inputs.size()==1 && outputs.size()>=1;
	}

}
