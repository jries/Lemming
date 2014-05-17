package org.lemming.tests;

import java.util.Map;
import java.util.Map.Entry;

import org.junit.Test;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.utils.Parameters;

import static org.junit.Assert.*;

public class ParameterTest {
	
	@Test
	public void testEntrySet() {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		Map<String,Object> parameters = Parameters.getParameterMap(d);
		
		for (Entry<String, Object> e : parameters.entrySet())
			System.out.println(e.getKey()+" = "+e.getValue());
		
		assertEquals(100l,parameters.get("noFrames"));
	}

	@Test
	public void testMap() throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		Map<String,Object> parameters = Parameters.getParameterMap(d);
		
		for(String parameter : parameters.keySet())
			System.out.println(parameter + " = " + parameters.get(parameter));
		
		assertEquals(100l, (long) Parameters.getParameter(d, "noFrames"));
	}
	
	@Test
	public void testSetParameter() throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		Map<String,Object> parameters = Parameters.getParameterMap(d);
		
		for(String parameter : parameters.keySet())
			System.out.println(parameter + " = " + parameters.get(parameter));
		
		parameters.put("noFrames", 50);
		
		for(String parameter : parameters.keySet())
			System.out.println(parameter + " = " + parameters.get(parameter));
		
		assertEquals(50l, (long) Parameters.getParameter(d, "noFrames"));
		
		assertEquals(50l, d.noFrames);
	}
	
	@Test
	public void testGetUnknownParameter() {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		Map<String,Object> params = Parameters.getParameterMap(d);
		
		Object o = params.get("nonono");
		
		if (o != null)
			fail();
	}
	
	@Test
	public void testSetUnknownParameter() {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		Map<String,Object> params = Parameters.getParameterMap(d);
		
		params.put("nonono",null);
		
		if (params.containsKey("nonono"))
			fail();
	}
	
	@Test
	public void testReadOnly() {
		DummyFrameProducer d = new DummyFrameProducer(100);
		
		try {
			assertTrue(Parameters.isReadOnly(d, "noFrames"));
		} catch (NoSuchFieldException | SecurityException
				| IllegalArgumentException | IllegalAccessException e) {
			fail();
		}
		
	}
}
