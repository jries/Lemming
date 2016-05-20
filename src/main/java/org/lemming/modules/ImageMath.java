package org.lemming.modules;

import java.util.Iterator;
import java.util.NoSuchElementException;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.SingleRunModule;

/**
 * a module for image calculations
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public class ImageMath<T extends RealType<T>> extends SingleRunModule {
	
	public enum operators {
		ADDITION, SUBSTRACTION, MULTIPLICATION, DIVISION, NONE
	}

	private operators operator;
	private int counter;
	private Store inputA;
	private Store inputB;
	private final int frames;
	
	public ImageMath(int frames){
		this.frames = frames;
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
		} catch (NoSuchElementException | NullPointerException ex){
			System.err.println("Input provided not correct!");
			Thread.currentThread().interrupt();
		}
	
		boolean loop = true;
		while(loop){									// check for minimum elements in the two input stores
			if ((frames < inputB.size()) && (frames < inputA.size()))
				loop = false;
			pause(10);
		}
		System.out.println("Image Math - Input ready");
		start = System.currentTimeMillis();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		Frame<T> frameB = (Frame<T>) data;
		if (frameB == null) { return null; }
		Frame<T> test = null;
		int maxFrames = Math.min(frames*4,Math.max(inputA.size(), inputB.size()));
		for(int i=0;i<maxFrames;i++){
			test = (Frame<T>)inputA.poll();
			if (test==null) continue;
			if (frameB.getFrameNumber()==test.getFrameNumber())
				break;
			try {
				inputA.put(test);
			} catch (InterruptedException e) {
				System.out.println(e.getMessage());
			}
		}
		Frame<T> frameA = test;
		try {
			if (frameA == null) {
				inputB.put(frameB);
				return null;
			}

			// if no match put it back to inputs
			if (frameA.getFrameNumber() != frameB.getFrameNumber()) {
				inputB.put(frameB);
				//inputA.put(frameA);
				return null;
			}

			Pair<Frame<T>, Frame<T>> framePair = new ValuePair<>(frameA, frameB);

			if (frameB.isLast()) { // make the poison pill
				ImgLib2Frame<T> lastFrame = process1(framePair);
				lastFrame.setLast(true);
				newOutput(lastFrame);
				cancel();
				counter++;
				return null;
			}

			newOutput(process1(framePair));
			counter++;
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}

		return null;
	}

	private ImgLib2Frame<T> process1(Pair<Frame<T>, Frame<T>> framePair) {
		
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
	            double val = cursorB.get().getRealDouble() - cursorA.get().getRealDouble();
	            val = val<0?0:val; 				// check for negative values
	            cursorA.get().setReal(val);
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
		
		return new ImgLib2Frame<>(framePair.getA().getFrameNumber(), framePair.getA().getWidth(), 
				framePair.getA().getHeight(), framePair.getA().getPixelDepth(), intervalA);
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Math done with " + counter + " frames in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==2 && outputs.size()>=1;
	}

}
